import argparse
import json
import math
import os
import pathlib
import random
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv


ARCH = "C"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def grad_scaler(device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def read_features(path, id_column):
    df = pd.read_csv(path)
    if id_column not in df.columns:
        df = df.rename(columns={df.columns[0]: id_column})
    df[id_column] = df[id_column].astype(str)
    feature_columns = [c for c in df.columns if c != id_column]
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[[id_column]].join(features), feature_columns


def read_trait(path):
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
    missing = [c for c in ["Env", "Hybrid", "Yield_Mg_ha"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["Env"] = df["Env"].astype(str)
    df["Hybrid"] = df["Hybrid"].astype(str)
    df["Yield_Mg_ha"] = pd.to_numeric(df["Yield_Mg_ha"], errors="coerce")
    return df


def feature_matrix(df, columns):
    x = df[columns].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def prepare_inputs(args):
    geno_df, geno_columns = read_features(args.genotypes, "Hybrid")
    env_df, env_columns = read_features(args.train_env, "Env")
    geno_scaler = RobustScaler()
    env_scaler = RobustScaler()
    geno_scaled = geno_scaler.fit_transform(feature_matrix(geno_df, geno_columns)).astype(np.float32)
    env_scaled = env_scaler.fit_transform(feature_matrix(env_df, env_columns)).astype(np.float32)
    n_components = min(args.pca, geno_scaled.shape[0], geno_scaled.shape[1])
    if n_components < 1:
        raise ValueError("Genotype table has no usable feature columns.")
    solver = "randomized" if n_components < min(geno_scaled.shape) else "full"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=args.seed)
    geno_reduced = pca.fit_transform(geno_scaled).astype(np.float32)
    return geno_df, env_df, geno_columns, env_columns, geno_scaler, env_scaler, pca, geno_reduced, env_scaled


def split_by_pair(df, fraction, seed):
    if len(df) < 2:
        raise ValueError("At least two training rows are required.")
    pairs = df[["Hybrid", "Env"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * fraction)))
    val_pairs = {tuple(x) for x in pairs.iloc[order[:n_val]].to_numpy()}
    mask = pd.Series([p in val_pairs for p in zip(df["Hybrid"], df["Env"])], index=df.index)
    train_df = df.loc[~mask].reset_index(drop=True)
    val_df = df.loc[mask].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        order = rng.permutation(len(df))
        n_val = max(1, int(round(len(df) * fraction)))
        val_idx = set(order[:n_val])
        mask = pd.Series([i in val_idx for i in range(len(df))], index=df.index)
        train_df = df.loc[~mask].reset_index(drop=True)
        val_df = df.loc[mask].reset_index(drop=True)
    return train_df, val_df


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


class GxEDataset(Dataset):
    def __init__(self, trait, geno_reduced, env_scaled, g2i, e2i, y_scaler, k):
        self.trait = trait.reset_index(drop=True)
        self.geno_reduced = geno_reduced
        self.env_scaled = env_scaled
        self.g2i = g2i
        self.e2i = e2i
        self.y_scaler = y_scaler
        self.k = k

    def __len__(self):
        return len(self.trait)

    def __getitem__(self, idx):
        row = self.trait.iloc[idx]
        graph = build_graph(self.geno_reduced[self.g2i[row["Hybrid"]]], self.env_scaled[self.e2i[row["Env"]]], self.k)
        y = float(row["Yield_Mg_ha"])
        y = float(self.y_scaler.transform([[y]])[0, 0])
        return graph, torch.tensor(y, dtype=torch.float32)


def collate(batch):
    graphs, y = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(list(y))


def pool_heads(hidden, heads):
    value = max(1, min(4, heads))
    while hidden % value != 0:
        value -= 1
    return value


class SuperNodePool(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 2, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.query)

    def forward(self, x, batch):
        out = []
        for graph_id in range(int(batch.max().item()) + 1):
            nodes = x[batch == graph_id].unsqueeze(0)
            query = self.query.expand(1, -1, -1)
            pooled, _ = self.attention(query, nodes, nodes)
            out.append(self.norm(pooled).reshape(-1))
        return torch.stack(out, dim=0)


class GxEGAT(nn.Module):
    def __init__(self, hidden, heads, dropout, rounds):
        super().__init__()
        self.value_proj = nn.Linear(1, hidden)
        self.type_embedding = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList(
            [GATv2Conv(hidden, hidden, heads=heads, concat=False, edge_dim=1, dropout=dropout) for _ in range(rounds)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.super_pool = SuperNodePool(hidden, pool_heads(hidden, heads), dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, 256),
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
        return self.mlp(self.super_pool(x, data.batch)).squeeze(-1)


def invert_y(values, scaler):
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def score(y_true, y_pred):
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else math.nan
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else math.nan
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "r2": r2, "n": int(len(y_true))}


def mixed_loss(pred, y, weight):
    return weight * F.mse_loss(pred, y) + (1.0 - weight) * F.l1_loss(pred, y)


def evaluate(model, loader, device, y_scaler):
    model.eval()
    pred_all, y_all = [], []
    with torch.no_grad():
        for graphs, y in loader:
            graphs = graphs.to(device)
            with autocast_for(device):
                pred = model(graphs)
            pred_all.append(pred.cpu().numpy())
            y_all.append(y.numpy())
    return score(invert_y(np.concatenate(y_all), y_scaler), invert_y(np.concatenate(pred_all), y_scaler))


def train_epoch(model, loader, optimizer, scaler, device, mse_weight):
    model.train()
    total = 0.0
    count = 0
    for graphs, y in loader:
        graphs = graphs.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device):
            pred = model(graphs)
            loss = mixed_loss(pred, y, mse_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * len(y)
        count += len(y)
    return total / max(1, count)


def make_loader(dataset, batch, workers, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Architecture C")
    parser.add_argument("--genotypes", default="Genotypes.csv")
    parser.add_argument("--train-trait", default="train_trait.csv")
    parser.add_argument("--train-env", default="train_env_vectors.csv")
    parser.add_argument("--outdir", default="results/arch_c")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.8)
    parser.add_argument("--pca", type=int, default=548)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    set_seed(args.seed)
    device = device_from(args.device)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geno_df, env_df, geno_columns, env_columns, geno_scaler, env_scaler, pca, geno_reduced, env_scaled = prepare_inputs(args)
    g2i = {h: i for i, h in enumerate(geno_df["Hybrid"])}
    e2i = {e: i for i, e in enumerate(env_df["Env"])}
    trait = read_trait(args.train_trait)
    trait = trait[trait["Hybrid"].isin(g2i) & trait["Env"].isin(e2i) & trait["Yield_Mg_ha"].notna()].reset_index(drop=True)
    if trait.empty:
        raise ValueError("No training rows match the genotype and environment tables.")
    train_df, val_df = split_by_pair(trait, args.val_fraction, args.seed)
    y_scaler = RobustScaler().fit(train_df[["Yield_Mg_ha"]].to_numpy(dtype=np.float32))

    train_loader = make_loader(GxEDataset(train_df, geno_reduced, env_scaled, g2i, e2i, y_scaler, args.k), args.batch, args.workers, True, device)
    val_loader = make_loader(GxEDataset(val_df, geno_reduced, env_scaled, g2i, e2i, y_scaler, args.k), args.batch, args.workers, False, device)
    model = GxEGAT(args.hidden, args.heads, args.dropout, args.rounds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 2))
    scaler = grad_scaler(device)
    best_score = -math.inf
    best_metrics = {}
    patience_left = args.patience
    history = []

    print(f"Architecture {ARCH}")
    print(f"Device: {device}")
    print(f"Genotype PCA explained variance: {float(np.sum(pca.explained_variance_ratio_)):.4f}")
    print(f"Train rows: {len(train_df)}  Validation rows: {len(val_df)}")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, args.mse_weight)
        val_metrics = evaluate(model, val_loader, device, y_scaler)
        scheduler.step(val_metrics["rmse"])
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"Epoch {epoch:03d} loss={loss:.5f} val_rmse={val_metrics['rmse']:.5f} val_pcc={val_metrics['pcc']:.5f}")
        model_score = val_metrics["pcc"] if math.isfinite(val_metrics["pcc"]) else -val_metrics["rmse"]
        if model_score > best_score:
            best_score = model_score
            best_metrics = val_metrics
            patience_left = args.patience
            torch.save(
                {
                    "architecture": ARCH,
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "geno_columns": geno_columns,
                    "env_columns": env_columns,
                    "geno_scaler": geno_scaler,
                    "env_scaler": env_scaler,
                    "pca": pca,
                    "y_scaler": y_scaler,
                    "hidden": args.hidden,
                    "heads": args.heads,
                    "dropout": args.dropout,
                    "rounds": args.rounds,
                    "k": args.k,
                },
                outdir / "best_model.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    with open(outdir / "best_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)


if __name__ == "__main__":
    main()
