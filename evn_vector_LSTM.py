"""Create fixed environment vectors from daily weather records with an LSTM.

The LSTM is trained as a separate environment-embedding step. The exported
vectors are then used as fixed environment node features by the GxE-GNN scripts.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


WEATHER_FEATURES = [
    "ALLSKY_SFC_PAR_TOT",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DNI",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build LSTM environment vectors from weather sequences.")
    parser.add_argument("--weather-csv", default="CSV_Files/Test_Weather.csv")
    parser.add_argument("--output", default="Weather/env_vectors21.csv")
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--model-in", default=None)
    parser.add_argument("--env-col", default="Env")
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--hidden-dim", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(state):
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var"], dtype=np.float64)
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def load_weather(path, env_col, date_col, scaler=None, fill_values=None):
    df = pd.read_csv(path)
    missing = [column for column in [env_col, date_col] + WEATHER_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing weather columns: {missing}")

    df = df.sort_values([env_col, date_col]).reset_index(drop=True)
    features = df[WEATHER_FEATURES].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        fill_values = features.median().fillna(0.0)
    else:
        fill_values = pd.Series(fill_values, index=WEATHER_FEATURES, dtype=np.float64)
    features = features.fillna(fill_values).fillna(0.0)

    values = features.to_numpy(dtype=np.float64)
    scaler = scaler or StandardScaler().fit(values)
    df.loc[:, WEATHER_FEATURES] = scaler.transform(values)

    envs, sequences = [], []
    for env, group in df.groupby(env_col, sort=True):
        envs.append(str(env))
        sequences.append(group[WEATHER_FEATURES].to_numpy(dtype=np.float32))

    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = np.zeros((len(sequences), int(lengths.max()), len(WEATHER_FEATURES)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence
    return envs, torch.tensor(padded), lengths, scaler, fill_values


class EnvironmentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        return hidden[-1]

    def forward(self, x, lengths):
        z = self.encode(x, lengths)
        repeated = z.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded), z


def masked_mse(pred, target, lengths):
    mask = torch.arange(target.size(1), device=target.device).unsqueeze(0) < lengths.unsqueeze(1)
    return ((pred - target).pow(2) * mask.unsqueeze(-1)).sum() / (mask.sum() * target.size(-1))


def train(model, x, lengths, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, _ = model(x, lengths)
        loss = masked_mse(reconstruction, x, lengths)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:03d} reconstruction_mse={loss.item():.6f}")
    model.load_state_dict(best_state)
    return best_loss


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    checkpoint = load_checkpoint(args.model_in, device) if args.model_in else None
    scaler = make_scaler(checkpoint["scaler"]) if checkpoint else None
    fill_values = checkpoint["fill_values"] if checkpoint else None
    hidden_dim = int(checkpoint["hidden_dim"]) if checkpoint else args.hidden_dim

    envs, x, lengths, scaler, fill_values = load_weather(args.weather_csv, args.env_col, args.date_col, scaler, fill_values)
    x, lengths = x.to(device), lengths.to(device)
    model = EnvironmentLSTM(len(WEATHER_FEATURES), hidden_dim).to(device)

    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        best_loss = train(model, x, lengths, args)
        print(f"best_reconstruction_mse={best_loss:.6f}")

    if args.model_out and not checkpoint:
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden_dim": hidden_dim,
                "fill_values": fill_values.to_dict(),
                "scaler": {"mean": scaler.mean_, "scale": scaler.scale_, "var": scaler.var_},
            },
            args.model_out,
        )

    model.eval()
    with torch.no_grad():
        vectors = model.encode(x, lengths).cpu().numpy()

    output = pd.DataFrame(vectors, index=envs, columns=[str(i) for i in range(vectors.shape[1])])
    output.index.name = "Env"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output)
    print(f"device={device}")
    print(f"saved={args.output} rows={len(output)} dims={vectors.shape[1]}")


if __name__ == "__main__":
    main()
