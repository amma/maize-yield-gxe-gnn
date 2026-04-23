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
from torch_geometric.nn import GATv2Conv, global_mean_pool


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
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


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def standardize_feature_table(path, id_column):
    df = pd.read_csv(path)
    if id_column not in df.columns:
        df = df.rename(columns={df.columns[0]: id_column})
    df[id_column] = df[id_column].astype(str)
    feature_columns = [c for c in df.columns if c != id_column]
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[[id_column]].join(features), feature_columns


def standardize_trait_table(path, require_yield):
    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    renames = {}
    for target, choices in {
        "Env": ["env", "environment", "site", "location"],
        "Hybrid": ["hybrid", "genotype", "hybrid_id", "entry"],
        "Yield_Mg_ha": ["yield_mg_ha", "yield", "yield_mg_ha_", "trait"],
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
    required = ["Env", "Hybrid"] + (["Yield_Mg_ha"] if require_yield else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["Env"] = df["Env"].astype(str)
    df["Hybrid"] = df["Hybrid"].astype(str)
    if "Yield_Mg_ha" in df.columns:
        df["Yield_Mg_ha"] = pd.to_numeric(df["Yield_Mg_ha"], errors="coerce")
    return df


def matrix_from_table(df, id_column, feature_columns):
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {id_column}: {missing[:10]}")
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def fit_preprocessors(genotype_path, train_env_path, pca_components, seed):
    geno_df, geno_columns = standardize_feature_table(genotype_path, "Hybrid")
    env_df, env_columns = standardize_feature_table(train_env_path, "Env")
    geno_raw = matrix_from_table(geno_df, "Hybrid", geno_columns)
    env_raw = matrix_from_table(env_df, "Env", env_columns)
    geno_scaler = RobustScaler()
    env_scaler = RobustScaler()
    geno_scaled = geno_scaler.fit_transform(geno_raw).astype(np.float32)
    env_scaled = env_scaler.fit_transform(env_raw).astype(np.float32)
    n_components = min(pca_components, geno_scaled.shape[0], geno_scaled.shape[1])
    min_dim = min(geno_scaled.shape[0], geno_scaled.shape[1])
    solver = "randomized" if n_components < min_dim else "full"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed)
    geno_reduced = pca.fit_transform(geno_scaled).astype(np.float32)
    return {
        "geno_df": geno_df,
        "env_df": env_df,
        "geno_columns": geno_columns,
        "env_columns": env_columns,
        "geno_scaler": geno_scaler,
        "env_scaler": env_scaler,
        "pca": pca,
        "geno_reduced": geno_reduced,
        "env_scaled": env_scaled,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def transform_genotypes(genotype_path, geno_columns, geno_scaler, pca):
    geno_df, _ = standardize_feature_table(genotype_path, "Hybrid")
    geno_raw = matrix_from_table(geno_df, "Hybrid", geno_columns)
    geno_scaled = geno_scaler.transform(geno_raw).astype(np.float32)
    return geno_df, pca.transform(geno_scaled).astype(np.float32)


def transform_environment(env_path, env_columns, env_scaler):
    env_df, _ = standardize_feature_table(env_path, "Env")
    env_raw = matrix_from_table(env_df, "Env", env_columns)
    return env_df, env_scaler.transform(env_raw).astype(np.float32)


def filter_trait(trait_df, g2i, e2i, require_yield):
    keep = trait_df["Hybrid"].isin(g2i) & trait_df["Env"].isin(e2i)
    if require_yield and "Yield_Mg_ha" in trait_df.columns:
        keep &= trait_df["Yield_Mg_ha"].notna()
    return trait_df.loc[keep].reset_index(drop=True)


def split_by_pair(trait_df, val_fraction, seed):
    pairs = trait_df[["Hybrid", "Env"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * val_fraction)))
    val_pairs = {tuple(x) for x in pairs.iloc[order[:n_val]].to_numpy()}
    pair_index = list(zip(trait_df["Hybrid"], trait_df["Env"]))
    val_mask = pd.Series([p in val_pairs for p in pair_index], index=trait_df.index)
    train_df = trait_df.loc[~val_mask].reset_index(drop=True)
    val_df = trait_df.loc[val_mask].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        idx = rng.permutation(len(trait_df))
        n_val = max(1, int(round(len(trait_df) * val_fraction)))
        val_idx = set(idx[:n_val])
        val_mask = pd.Series([i in val_idx for i in range(len(trait_df))])
        train_df = trait_df.loc[~val_mask].reset_index(drop=True)
        val_df = trait_df.loc[val_mask].reset_index(drop=True)
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
        neighbors = np.argpartition(sims[i], -k)[-k:]
        for j in neighbors:
            if np.isfinite(sims[i, j]):
                rows.append(offset + i)
                cols.append(offset + j)
                weights.append(sims[i, j])
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def build_graph(g_vec, e_vec, architecture, k):
    ng = len(g_vec)
    ne = len(e_vec)
    g_nodes = np.arange(ng, dtype=np.int64)
    e_nodes = np.arange(ng, ng + ne, dtype=np.int64)
    src_ge = np.repeat(g_nodes, ne)
    dst_ge = np.tile(e_nodes, ng)
    src = [src_ge, dst_ge]
    dst = [dst_ge, src_ge]
    weights = [np.ones(len(src_ge), dtype=np.float32), np.ones(len(src_ge), dtype=np.float32)]
    if architecture in {"B", "C"}:
        gg_src, gg_dst, gg_w = topk_edges(g_vec, 0, k)
        ee_src, ee_dst, ee_w = topk_edges(e_vec, ng, k)
        src.extend([gg_src, ee_src])
        dst.extend([gg_dst, ee_dst])
        weights.extend([gg_w, ee_w])
    edge_index = torch.tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long)
    edge_attr = torch.tensor(np.concatenate(weights)[:, None], dtype=torch.float32)
    node_values = np.concatenate([g_vec, e_vec]).astype(np.float32)
    node_type = np.concatenate([np.zeros(ng, dtype=np.int64), np.ones(ne, dtype=np.int64)])
    return Data(
        x=torch.tensor(node_values[:, None], dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=torch.tensor(node_type, dtype=torch.long),
    )


class GxEDataset(Dataset):
    def __init__(self, trait_df, geno_reduced, env_scaled, g2i, e2i, architecture, k, y_scaler=None, include_y=True):
        self.trait = trait_df.reset_index(drop=True)
        self.geno_reduced = geno_reduced
        self.env_scaled = env_scaled
        self.g2i = g2i
        self.e2i = e2i
        self.architecture = architecture
        self.k = k
        self.y_scaler = y_scaler
        self.include_y = include_y

    def __len__(self):
        return len(self.trait)

    def __getitem__(self, idx):
        row = self.trait.iloc[idx]
        hybrid = row["Hybrid"]
        env = row["Env"]
        graph = build_graph(
            self.geno_reduced[self.g2i[hybrid]],
            self.env_scaled[self.e2i[env]],
            self.architecture,
            self.k,
        )
        if self.include_y and "Yield_Mg_ha" in row and pd.notna(row["Yield_Mg_ha"]):
            y_value = float(row["Yield_Mg_ha"])
            if self.y_scaler is not None:
                y_value = float(self.y_scaler.transform([[y_value]])[0, 0])
            y = torch.tensor(y_value, dtype=torch.float32)
        else:
            y = torch.tensor(float("nan"), dtype=torch.float32)
        return graph, y, hybrid, env


def collate_graphs(batch):
    graphs, y, hybrids, envs = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(list(y)), list(hybrids), list(envs)


class SuperNodePool(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.query)

    def forward(self, x, batch):
        outputs = []
        graph_count = int(batch.max().item()) + 1
        for graph_id in range(graph_count):
            nodes = x[batch == graph_id].unsqueeze(0)
            query = self.query.expand(1, -1, -1)
            pooled, _ = self.attention(query, nodes, nodes)
            outputs.append(self.norm(pooled.squeeze(0).squeeze(0)))
        return torch.stack(outputs, dim=0)


class GxEGAT(nn.Module):
    def __init__(self, architecture, hidden=128, heads=8, dropout=0.25):
        super().__init__()
        self.architecture = architecture.upper()
        self.value_proj = nn.Linear(1, hidden)
        self.type_embedding = nn.Embedding(2, hidden)
        self.conv1 = GATv2Conv(hidden, hidden, heads=heads, edge_dim=1, dropout=dropout)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=2, edge_dim=1, dropout=dropout)
        self.conv3 = GATv2Conv(hidden * 2, hidden, heads=1, concat=False, edge_dim=1, dropout=dropout)
        pool_heads = max(1, min(4, heads))
        while hidden % pool_heads != 0:
            pool_heads -= 1
        self.super_pool = SuperNodePool(hidden, pool_heads, dropout) if self.architecture == "C" else None
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(1, hidden // 2)),
            nn.LayerNorm(max(1, hidden // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden // 2), 1),
        )

    def forward(self, data):
        x = self.value_proj(data.x) + self.type_embedding(data.node_type)
        x = F.leaky_relu(self.conv1(x, data.edge_index, data.edge_attr), 0.2)
        x = F.leaky_relu(self.conv2(x, data.edge_index, data.edge_attr), 0.2)
        x = F.leaky_relu(self.conv3(x, data.edge_index, data.edge_attr), 0.2)
        if self.architecture == "C":
            pooled = self.super_pool(x, data.batch)
        else:
            mask = data.node_type == 0
            pooled = global_mean_pool(x[mask], data.batch[mask])
        return self.mlp(pooled).squeeze(-1)


def mixed_loss(pred, target, mse_weight):
    return mse_weight * F.mse_loss(pred, target) + (1.0 - mse_weight) * F.l1_loss(pred, target)


def invert_y(values, scaler):
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if scaler is None:
        return values.ravel()
    return scaler.inverse_transform(values).ravel()


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
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0 else math.nan
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else math.nan
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "r2": r2, "bias": bias, "n": int(len(y_true))}


def evaluate(model, loader, device, y_scaler=None):
    model.eval()
    pred_scaled, true_scaled, hybrids, envs = [], [], [], []
    with torch.no_grad():
        for graphs, y, h, e in loader:
            graphs = graphs.to(device)
            with autocast_for(device):
                pred = model(graphs)
            pred_scaled.append(pred.detach().cpu().numpy())
            true_scaled.append(y.detach().cpu().numpy())
            hybrids.extend(h)
            envs.extend(e)
    pred_scaled = np.concatenate(pred_scaled) if pred_scaled else np.asarray([])
    true_scaled = np.concatenate(true_scaled) if true_scaled else np.asarray([])
    pred = invert_y(pred_scaled, y_scaler)
    true = invert_y(true_scaled, y_scaler)
    return true, pred, hybrids, envs, metrics(true, pred)


def train_epoch(model, loader, optimizer, scaler, device, mse_weight):
    model.train()
    total = 0.0
    count = 0
    for graphs, y, _, _ in loader:
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


def make_loader(dataset, batch_size, workers, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_graphs,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def train_main(architecture):
    architecture = architecture.upper()
    args = train_parser(architecture).parse_args()
    set_seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    device = select_device(args.device)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prep = fit_preprocessors(args.genotypes, args.train_env, args.pca, args.seed)
    g2i = {h: i for i, h in enumerate(prep["geno_df"]["Hybrid"])}
    e2i = {e: i for i, e in enumerate(prep["env_df"]["Env"])}
    trait = standardize_trait_table(args.train_trait, require_yield=True)
    trait = filter_trait(trait, g2i, e2i, require_yield=True)
    if trait.empty:
        raise ValueError("No training rows match the genotype and environment tables.")
    train_df, val_df = split_by_pair(trait, args.val_fraction, args.seed)
    y_scaler = RobustScaler().fit(train_df[["Yield_Mg_ha"]].to_numpy(dtype=np.float32))
    train_ds = GxEDataset(train_df, prep["geno_reduced"], prep["env_scaled"], g2i, e2i, architecture, args.k, y_scaler, True)
    val_ds = GxEDataset(val_df, prep["geno_reduced"], prep["env_scaled"], g2i, e2i, architecture, args.k, y_scaler, True)
    train_loader = make_loader(train_ds, args.batch, args.workers, True, device)
    val_loader = make_loader(val_ds, args.batch, args.workers, False, device)
    model = GxEGAT(architecture, args.hidden, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 2))
    scaler = grad_scaler(device)
    best_score = -math.inf
    best_metrics = None
    patience_left = args.patience
    history = []
    print(f"Architecture {architecture}")
    print(f"Device: {device}")
    print(f"Genotype PCA explained variance: {prep['explained_variance']:.4f}")
    print(f"Train rows: {len(train_df)}  Validation rows: {len(val_df)}")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, args.mse_weight)
        _, _, _, _, val_metrics = evaluate(model, val_loader, device, y_scaler)
        scheduler.step(val_metrics["rmse"] if math.isfinite(val_metrics["rmse"]) else loss)
        score = val_metrics["pcc"] if math.isfinite(val_metrics["pcc"]) else -val_metrics["rmse"]
        row = {"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:03d} loss={loss:.5f} val_rmse={val_metrics['rmse']:.5f} val_pcc={val_metrics['pcc']:.5f}")
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            patience_left = args.patience
            checkpoint = {
                "architecture": architecture,
                "model_state": model.state_dict(),
                "config": vars(args),
                "geno_columns": prep["geno_columns"],
                "env_columns": prep["env_columns"],
                "geno_scaler": prep["geno_scaler"],
                "env_scaler": prep["env_scaler"],
                "pca": prep["pca"],
                "y_scaler": y_scaler,
                "hidden": args.hidden,
                "heads": args.heads,
                "dropout": args.dropout,
                "k": args.k,
                "explained_variance": prep["explained_variance"],
            }
            torch.save(checkpoint, outdir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    save_json(outdir / "best_metrics.json", best_metrics or {})


def test_main(architecture):
    architecture = architecture.upper()
    args = test_parser(architecture).parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch_load(args.model_path, device)
    checkpoint_arch = checkpoint.get("architecture", architecture)
    model = GxEGAT(checkpoint_arch, checkpoint["hidden"], checkpoint["heads"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    geno_df, geno_reduced = transform_genotypes(
        args.genotypes,
        checkpoint["geno_columns"],
        checkpoint["geno_scaler"],
        checkpoint["pca"],
    )
    env_df, env_scaled = transform_environment(
        args.test_env,
        checkpoint["env_columns"],
        checkpoint["env_scaler"],
    )
    g2i = {h: i for i, h in enumerate(geno_df["Hybrid"])}
    e2i = {e: i for i, e in enumerate(env_df["Env"])}
    trait = standardize_trait_table(args.test_trait, require_yield=False)
    trait = filter_trait(trait, g2i, e2i, require_yield=False)
    if trait.empty:
        raise ValueError("No test rows match the genotype and environment tables.")
    include_y = "Yield_Mg_ha" in trait.columns and trait["Yield_Mg_ha"].notna().any()
    dataset = GxEDataset(trait, geno_reduced, env_scaled, g2i, e2i, checkpoint_arch, checkpoint["k"], checkpoint.get("y_scaler"), include_y)
    loader = make_loader(dataset, args.batch, args.workers, False, device)
    true, pred, hybrids, envs, test_metrics = evaluate(model, loader, device, checkpoint.get("y_scaler"))
    predictions = pd.DataFrame({"Env": envs, "Hybrid": hybrids, "Predicted_Yield_Mg_ha": pred})
    if include_y:
        predictions["Observed_Yield_Mg_ha"] = true
    predictions.to_csv(outdir / "predictions.csv", index=False)
    if args.ground_truth:
        truth = standardize_trait_table(args.ground_truth, require_yield=True)
        merged = predictions.merge(truth[["Env", "Hybrid", "Yield_Mg_ha"]], on=["Env", "Hybrid"], how="inner")
        test_metrics = metrics(merged["Yield_Mg_ha"], merged["Predicted_Yield_Mg_ha"])
    save_json(outdir / "metrics.json", test_metrics)
    print(f"Architecture {checkpoint_arch}")
    print(f"Device: {device}")
    print(f"Rows predicted: {len(predictions)}")
    print(f"RMSE: {test_metrics['rmse']:.5f}  PCC: {test_metrics['pcc']:.5f}  R2: {test_metrics['r2']:.5f}")


def train_parser(architecture):
    lower = architecture.lower()
    parser = argparse.ArgumentParser(description=f"Train Architecture {architecture}")
    parser.add_argument("--genotypes", default="Genotypes.csv")
    parser.add_argument("--train-trait", default="train_trait.csv")
    parser.add_argument("--train-env", default="train_env_vectors.csv")
    parser.add_argument("--outdir", default=f"results/arch_{lower}")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.8)
    parser.add_argument("--pca", type=int, default=548)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def test_parser(architecture):
    lower = architecture.lower()
    parser = argparse.ArgumentParser(description=f"Test Architecture {architecture}")
    parser.add_argument("--model-path", default=f"results/arch_{lower}/best_model.pt")
    parser.add_argument("--genotypes", default="Genotypes.csv")
    parser.add_argument("--test-trait", default="test_trait.csv")
    parser.add_argument("--test-env", default="test_env_vectors.csv")
    parser.add_argument("--ground-truth", default="")
    parser.add_argument("--outdir", default=f"results/arch_{lower}_test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser
