"""Create fixed environment vectors from daily weather records with an LSTM.

The script groups weather observations by environment, trains a compact LSTM
sequence autoencoder, and writes one learned summary vector per environment for
the GxE-GNN training and test scripts.
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
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURES = [
    "ALLSKY_SFC_PAR_TOT",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DNI",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build LSTM environment vectors from daily weather data.")
    parser.add_argument("--weather-csv", default="CSV_Files/Test_Weather.csv")
    parser.add_argument("--output", default="Weather/env_vectors21.csv")
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--model-in", default=None)
    parser.add_argument("--env-col", default="Env")
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=21)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def choose_device(cpu):
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def resolve_features(df, args):
    if args.features:
        missing = [column for column in args.features if column not in df.columns]
        if missing:
            raise ValueError(f"Weather file is missing requested feature columns: {missing}")
        return args.features

    if all(column in df.columns for column in DEFAULT_FEATURES):
        return DEFAULT_FEATURES

    excluded = {args.env_col, args.date_col}
    numeric_columns = []
    for column in df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            numeric_columns.append(column)

    if not numeric_columns:
        raise ValueError("No numeric weather feature columns were found.")
    return numeric_columns


def scaler_from_checkpoint(checkpoint):
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = np.asarray(checkpoint["scaler_var"], dtype=np.float64)
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def load_weather_sequences(path, args, feature_columns=None, scaler=None, fill_values=None):
    df = pd.read_csv(path)
    if args.env_col not in df.columns:
        raise ValueError(f"Weather file is missing the environment column '{args.env_col}'.")

    feature_columns = feature_columns or resolve_features(df, args)
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Weather file is missing expected feature columns: {missing}")

    sort_columns = [args.env_col] + ([args.date_col] if args.date_col in df.columns else [])
    df = df.sort_values(sort_columns).reset_index(drop=True)

    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = features.median().fillna(0.0)
    else:
        fill_values = pd.Series(fill_values, index=feature_columns, dtype=np.float64)
    features = features.fillna(fill_values).fillna(0.0)
    feature_array = features.to_numpy(dtype=np.float64)

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_array)
    else:
        scaled = scaler.transform(feature_array)
    df.loc[:, feature_columns] = scaled

    env_names = []
    sequences = []
    for env, frame in df.groupby(args.env_col, sort=True):
        values = frame[feature_columns].to_numpy(dtype=np.float32)
        if len(values) == 0:
            continue
        env_names.append(str(env))
        sequences.append(values)

    if not sequences:
        raise ValueError("No environment sequences were created from the weather file.")

    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    max_len = int(lengths.max())
    padded = np.zeros((len(sequences), max_len, len(feature_columns)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence

    return env_names, feature_columns, scaler, fill_values, torch.tensor(padded), torch.tensor(lengths)


class LSTMEnvironmentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        return hidden[-1]

    def forward(self, x, lengths):
        embedding = self.encode(x, lengths)
        repeated = embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded), embedding


def masked_mse(prediction, target, lengths):
    time = torch.arange(target.size(1), device=target.device).unsqueeze(0)
    mask = time < lengths.unsqueeze(1)
    squared_error = (prediction - target).pow(2) * mask.unsqueeze(-1)
    denominator = mask.sum().clamp_min(1) * target.size(-1)
    return squared_error.sum() / denominator


def train_model(model, sequences, lengths, args, device):
    dataset = TensorDataset(sequences, lengths)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch_x, batch_lengths in loader:
            batch_x = batch_x.to(device)
            batch_lengths = batch_lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch_x, batch_lengths)
            loss = masked_mse(reconstruction, batch_x, batch_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_items += len(batch_x)

        epoch_loss = total_loss / max(1, total_items)
        if epoch_loss < best_loss - 1e-7:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:03d} reconstruction_mse={epoch_loss:.6f}")

        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path, model, args, feature_columns, scaler, fill_values):
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_var": scaler.var_,
        "fill_values": fill_values.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def export_vectors(model, env_names, sequences, lengths, output_path, device):
    dataset = TensorDataset(sequences, lengths)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    embeddings = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_lengths in loader:
            batch_x = batch_x.to(device)
            batch_lengths = batch_lengths.to(device)
            embeddings.append(model.encode(batch_x, batch_lengths).cpu().numpy())

    matrix = np.vstack(embeddings)
    columns = [str(i) for i in range(matrix.shape[1])]
    output = pd.DataFrame(matrix, index=env_names, columns=columns)
    output.index.name = "Env"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path)
    return output_path, output.shape


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.cpu)

    checkpoint = load_checkpoint(args.model_in, device) if args.model_in else None
    checkpoint_features = checkpoint["feature_columns"] if checkpoint else None
    checkpoint_scaler = scaler_from_checkpoint(checkpoint) if checkpoint else None
    checkpoint_fill = checkpoint["fill_values"] if checkpoint else None

    env_names, feature_columns, scaler, fill_values, sequences, lengths = load_weather_sequences(
        args.weather_csv,
        args,
        feature_columns=checkpoint_features,
        scaler=checkpoint_scaler,
        fill_values=checkpoint_fill,
    )

    if checkpoint:
        args.hidden_dim = int(checkpoint["hidden_dim"])
        args.num_layers = int(checkpoint["num_layers"])
        args.dropout = float(checkpoint["dropout"])

    model = LSTMEnvironmentEncoder(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"])

    sequences = sequences.to(device)
    lengths = lengths.to(device)
    best_loss = None if checkpoint else train_model(model, sequences, lengths, args, device)

    if args.model_out and not checkpoint:
        saved_model = save_checkpoint(args.model_out, model, args, feature_columns, scaler, fill_values)
        print(f"model_saved={saved_model}")

    output_path, shape = export_vectors(model, env_names, sequences, lengths, args.output, device)

    print(f"device={device}")
    print(f"features={','.join(feature_columns)}")
    if best_loss is not None:
        print(f"best_reconstruction_mse={best_loss:.6f}")
    print(f"saved={output_path} rows={shape[0]} dims={shape[1]}")


if __name__ == "__main__":
    main()
