import argparse
import json
import math
import pathlib
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


ARCH = "B"


def device_from(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_for(device):
    if device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except TypeError:
        return torch.cuda.amp.autocast()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def read_features(path, id_column):
    df = pd.read_csv(path)
    if id_column not in df.columns:
        df = df.rename(columns={df.columns[0]: id_column})
    df[id_column] = df[id_column].astype(str)
    features = [c for c in df.columns if c != id_column]
    x = df[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[[id_column]].join(x)


def read_trait(path, require_yield):
    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    renames = {}
    for target, choices in {
        "Env": ["env", "environment", "site", "location"],
        "Hybrid": ["hybrid", "genotype", "hybrid_id", "entry"],
        "Yield_Mg_ha": ["yield_mg_ha", "yield", "trait"],
    }.items():
        if target in df.columns:
            continue
        for choice in choices:
            if choice in lower:
                renames[lower[choice]] = target
                break
    df = df.rename(columns=renames)
    if not {"Env", "Hybrid"}.issubset(df.columns) and len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: "Env", df.columns[1]: "Hybrid"})
    if "Yield_Mg_ha" not in df.columns and len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]: "Yield_Mg_ha"})
    missing = [c for c in ["Env", "Hybrid"] + (["Yield_Mg_ha"] if require_yield else []) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["Env"] = df["Env"].astype(str)
    df["Hybrid"] = df["Hybrid"].astype(str)
    if "Yield_Mg_ha" in df.columns:
        df["Yield_Mg_ha"] = pd.to_numeric(df["Yield_Mg_ha"], errors="coerce")
    return df


def matrix(df, columns):
    x = df[columns].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def topk_edges(values, offset, k):
    n = len(values)
    if n <= 1 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    k = min(k, n - 1)
    diffs = np.abs(values[:, None] - values[None, :])
    sims = 1.0 / (1.0 + diffs)
    np.fill_diagonal(sims, -np.inf)
    rows, cols, weights = [], [], []
    for i in range(n):
        for j in np.argpartition(sims[i], -k)[-k:]:
            if np.isfinite(sims[i, j]):
                rows.append(offset + i)
                cols.append(offset + j)
                weights.append(sims[i, j])
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def build_graph(g_vec, e_vec, k):
    ng = len(g_vec)
    ne = len(e_vec)
    g_nodes = np.arange(ng, dtype=np.int64)
    e_nodes = np.arange(ng, ng + ne, dtype=np.int64)
    src_ge = np.repeat(g_nodes, ne)
    dst_ge = np.tile(e_nodes, ng)
    src = [src_ge, dst_ge]
    dst = [dst_ge, src_ge]
    weights = [np.ones(len(src_ge), dtype=np.float32), np.ones(len(src_ge), dtype=np.float32)]
    gg_src, gg_dst, gg_weight = topk_edges(g_vec, 0, k)
    ee_src, ee_dst, ee_weight = topk_edges(e_vec, ng, k)
    src.extend([gg_src, ee_src])
    dst.extend([gg_dst, ee_dst])
    weights.extend([gg_weight, ee_weight])
    edge_index = torch.tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long)
    edge_attr = torch.tensor(np.concatenate(weights)[:, None], dtype=torch.float32)
    values = np.concatenate([g_vec, e_vec]).astype(np.float32)
    node_type = np.concatenate([np.zeros(ng, dtype=np.int64), np.ones(ne, dtype=np.int64)])
    return Data(
        x=torch.tensor(values[:, None], dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=torch.tensor(node_type, dtype=torch.long),
    )


class GxETestDataset(Dataset):
    def __init__(self, trait, geno_reduced, env_scaled, g2i, e2i, k):
        self.trait = trait.reset_index(drop=True)
        self.geno_reduced = geno_reduced
        self.env_scaled = env_scaled
        self.g2i = g2i
        self.e2i = e2i
        self.k = k

    def __len__(self):
        return len(self.trait)

    def __getitem__(self, idx):
        row = self.trait.iloc[idx]
        graph = build_graph(self.geno_reduced[self.g2i[row["Hybrid"]]], self.env_scaled[self.e2i[row["Env"]]], self.k)
        y = float(row["Yield_Mg_ha"]) if "Yield_Mg_ha" in row and pd.notna(row["Yield_Mg_ha"]) else float("nan")
        return graph, torch.tensor(y, dtype=torch.float32), row["Hybrid"], row["Env"]


def collate(batch):
    graphs, y, hybrids, envs = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(list(y)), list(hybrids), list(envs)


class GxEGAT(nn.Module):
    def __init__(self, hidden, heads, dropout, rounds):
        super().__init__()
        self.value_proj = nn.Linear(1, hidden)
        self.type_embedding = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList(
            [GATv2Conv(hidden, hidden, heads=heads, concat=False, edge_dim=1, dropout=dropout) for _ in range(rounds)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = self.value_proj(data.x) + self.type_embedding(data.node_type)
        for layer, norm in zip(self.layers, self.norms):
            h = F.leaky_relu(layer(x, data.edge_index, data.edge_attr), 0.2)
            x = norm(x + h)
        g_mask = data.node_type == 0
        pooled = global_mean_pool(x[g_mask], data.batch[g_mask])
        return self.mlp(pooled).squeeze(-1)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if len(y_true) == 0:
        return {"rmse": math.nan, "mae": math.nan, "pcc": math.nan, "r2": math.nan, "bias": math.nan, "n": 0}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else math.nan
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else math.nan
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "r2": r2, "bias": float(np.mean(y_pred - y_true)), "n": int(len(y_true))}


def main():
    parser = argparse.ArgumentParser(description="Test Architecture B")
    parser.add_argument("--model-path", default="results/arch_b/best_model.pt")
    parser.add_argument("--genotypes", default="Genotypes.csv")
    parser.add_argument("--test-trait", default="test_trait.csv")
    parser.add_argument("--test-env", default="test_env_vectors.csv")
    parser.add_argument("--ground-truth", default="")
    parser.add_argument("--outdir", default="results/arch_b_test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = device_from(args.device)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.model_path, device)

    model = GxEGAT(checkpoint["hidden"], checkpoint["heads"], checkpoint["dropout"], checkpoint.get("rounds", 30)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    geno_df = read_features(args.genotypes, "Hybrid")
    env_df = read_features(args.test_env, "Env")
    geno_scaled = checkpoint["geno_scaler"].transform(matrix(geno_df, checkpoint["geno_columns"])).astype(np.float32)
    env_scaled = checkpoint["env_scaler"].transform(matrix(env_df, checkpoint["env_columns"])).astype(np.float32)
    geno_reduced = checkpoint["pca"].transform(geno_scaled).astype(np.float32)
    g2i = {h: i for i, h in enumerate(geno_df["Hybrid"])}
    e2i = {e: i for i, e in enumerate(env_df["Env"])}
    trait = read_trait(args.test_trait, require_yield=False)
    trait = trait[trait["Hybrid"].isin(g2i) & trait["Env"].isin(e2i)].reset_index(drop=True)
    if trait.empty:
        raise ValueError("No test rows match the genotype and environment tables.")

    loader = DataLoader(
        GxETestDataset(trait, geno_reduced, env_scaled, g2i, e2i, checkpoint["k"]),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    rows = []
    with torch.no_grad():
        for graphs, y, hybrids, envs in loader:
            graphs = graphs.to(device)
            with autocast_for(device):
                pred_scaled = model(graphs).cpu().numpy()
            pred = checkpoint["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            for env, hybrid, pred_value, obs in zip(envs, hybrids, pred, y.numpy()):
                row = {"Env": env, "Hybrid": hybrid, "Predicted_Yield_Mg_ha": pred_value}
                if np.isfinite(obs):
                    row["Observed_Yield_Mg_ha"] = checkpoint["y_scaler"].inverse_transform([[obs]])[0, 0]
                rows.append(row)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(outdir / "predictions.csv", index=False)

    if args.ground_truth:
        truth = read_trait(args.ground_truth, require_yield=True)
        merged = predictions.merge(truth[["Env", "Hybrid", "Yield_Mg_ha"]], on=["Env", "Hybrid"], how="inner")
        result = metrics(merged["Yield_Mg_ha"], merged["Predicted_Yield_Mg_ha"])
    elif "Observed_Yield_Mg_ha" in predictions.columns:
        result = metrics(predictions["Observed_Yield_Mg_ha"], predictions["Predicted_Yield_Mg_ha"])
    else:
        result = metrics([], [])
    with open(outdir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Architecture {ARCH}")
    print(f"Device: {device}")
    print(f"Rows predicted: {len(predictions)}")
    print(f"RMSE: {result['rmse']:.5f}  PCC: {result['pcc']:.5f}  R2: {result['r2']:.5f}")


if __name__ == "__main__":
    main()
