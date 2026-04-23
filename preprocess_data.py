"""Preprocess raw genotype, weather, and trait files for the GxE-GNN pipeline.

This script consolidates the original data-cleaning utilities: VCF-to-marker
CSV extraction, stochastic genotype imputation, low-variance marker filtering,
windowed marker similarity pruning, weather feature selection, and trait-table
cleanup.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_FEATURES = [
    "ALLSKY_SFC_PAR_TOT",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DNI",
]


def make_unique(values):
    seen = {}
    names = []
    for value in values:
        base = str(value)
        count = seen.get(base, 0)
        names.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1
    return names


def read_csv(path):
    return pd.read_csv(path, low_memory=False)


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def detect_format(path, explicit_format):
    if explicit_format != "auto":
        return explicit_format
    suffixes = "".join(Path(path).suffixes).lower()
    return "vcf" if suffixes.endswith(".vcf") or suffixes.endswith(".vcf.gz") else "csv"


def vcf_to_dataframe(path, id_col):
    try:
        import allel
    except ImportError as exc:
        raise ImportError("VCF preprocessing requires scikit-allel. Install it with `pip install scikit-allel`.") from exc

    callset = allel.read_vcf(path)
    gt = allel.GenotypeArray(callset["calldata/GT"])
    genotype = gt.to_n_alt(fill=-1).astype(np.int16)
    genotype = np.where(genotype < 0, -1, genotype + 1).T

    markers = make_unique(callset["variants/POS"])
    df = pd.DataFrame(genotype, columns=markers)
    df.insert(0, id_col, [str(sample) for sample in callset["samples"]])
    return df


def normalize_id_column(df, id_col):
    if id_col not in df.columns:
        df = df.rename(columns={df.columns[0]: id_col})
    df[id_col] = df[id_col].astype(str)
    return df


def marker_columns(df, id_col):
    return [column for column in df.columns if column != id_col]


def numeric_markers(df, id_col):
    markers = marker_columns(df, id_col)
    numeric = df[markers].apply(pd.to_numeric, errors="coerce")
    return numeric


def impute_column(values, rng, missing_value):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    missing = np.isnan(values) | (values == missing_value)
    observed = values[~missing]
    observed = observed[np.isin(observed, [1, 2, 3])]

    if observed.size == 0:
        probabilities = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    else:
        counts = np.asarray([(observed == value).sum() for value in [1, 2, 3]], dtype=np.float64)
        probabilities = counts / counts.sum()

    if missing.any():
        values[missing] = rng.choice([1, 2, 3], size=int(missing.sum()), p=probabilities)
    return values.astype(np.int16)


def impute_markers(markers, seed, missing_value):
    rng = np.random.default_rng(seed)
    imputed = {
        column: impute_column(markers[column], rng, missing_value)
        for column in markers.columns
    }
    return pd.DataFrame(imputed, index=markers.index)


def drop_low_variance_markers(markers, threshold):
    keep = []
    for column in markers.columns:
        frequency = markers[column].value_counts(normalize=True, dropna=False)
        if frequency.empty or frequency.max() < threshold:
            keep.append(column)
    return markers[keep]


def prune_similarity_window(markers, threshold):
    remaining = list(markers.columns)
    keep = []
    values = {column: markers[column].to_numpy(copy=False) for column in remaining}

    while remaining:
        reference = remaining[0]
        keep.append(reference)
        reference_values = values[reference]
        next_remaining = []
        for column in remaining[1:]:
            similarity = np.mean(values[column] == reference_values)
            if similarity < threshold:
                next_remaining.append(column)
        remaining = next_remaining

    return keep


def prune_similar_markers(markers, threshold, window_size):
    keep = []
    columns = list(markers.columns)
    for start in range(0, len(columns), window_size):
        window = columns[start : start + window_size]
        keep.extend(prune_similarity_window(markers[window], threshold))
    return markers[keep]


def preprocess_genotype(args):
    input_format = detect_format(args.input, args.input_format)
    if input_format == "vcf":
        df = vcf_to_dataframe(args.input, args.id_col)
    else:
        df = normalize_id_column(read_csv(args.input), args.id_col)

    ids = df[[args.id_col]].copy()
    markers = numeric_markers(df, args.id_col)
    initial_markers = markers.shape[1]

    if not args.skip_imputation:
        markers = impute_markers(markers, args.seed, args.missing_value)

    after_imputation = markers.shape[1]
    if not args.skip_low_variance:
        markers = drop_low_variance_markers(markers, args.low_variance_threshold)

    after_low_variance = markers.shape[1]
    if not args.skip_similarity_pruning:
        markers = prune_similar_markers(markers, args.similarity_threshold, args.window_size)

    output = pd.concat([ids.reset_index(drop=True), markers.reset_index(drop=True)], axis=1)
    output_path = write_csv(output, args.output)

    print(f"genotype_input={args.input}")
    print(f"genotype_output={output_path}")
    print(f"samples={len(output)}")
    print(f"markers_initial={initial_markers}")
    print(f"markers_after_imputation={after_imputation}")
    print(f"markers_after_low_variance={after_low_variance}")
    print(f"markers_final={markers.shape[1]}")


def clean_weather(args):
    df = read_csv(args.input)
    features = args.features or WEATHER_FEATURES
    required = [args.env_col, args.date_col] + features
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Weather file is missing required columns: {missing}")

    output = df[required].copy()
    output[args.env_col] = output[args.env_col].astype(str)
    for feature in features:
        output[feature] = pd.to_numeric(output[feature], errors="coerce")

    if args.missing == "drop":
        output = output.dropna(subset=required)
    elif args.missing == "median":
        output[features] = output[features].fillna(output[features].median()).fillna(0.0)
        output = output.dropna(subset=[args.env_col, args.date_col])
    elif args.missing == "zero":
        output[features] = output[features].fillna(0.0)
        output = output.dropna(subset=[args.env_col, args.date_col])

    output_path = write_csv(output, args.output)
    print(f"weather_input={args.input}")
    print(f"weather_output={output_path}")
    print(f"rows={len(output)}")
    print(f"features={','.join(features)}")


def find_column(df, preferred, candidates):
    if preferred in df.columns:
        return preferred
    lower = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find required column. Tried: {[preferred] + candidates}")


def clean_trait(args):
    df = read_csv(args.input)
    env_col = find_column(df, args.env_col, ["environment", "site", "location"])
    hybrid_col = find_column(df, args.hybrid_col, ["genotype", "hybrid_id", "entry"])
    yield_col = find_column(df, args.yield_col, ["yield", "trait", "yield_mg_ha"])

    output = df[[env_col, hybrid_col, yield_col]].copy()
    output.columns = ["Env", "Hybrid", "Yield_Mg_ha"]
    output["Env"] = output["Env"].astype(str)
    output["Hybrid"] = output["Hybrid"].astype(str)
    output["Yield_Mg_ha"] = pd.to_numeric(output["Yield_Mg_ha"], errors="coerce")

    required = ["Env", "Hybrid"] if args.allow_missing_yield else ["Env", "Hybrid", "Yield_Mg_ha"]
    output = output.dropna(subset=required)

    output_path = write_csv(output, args.output)
    print(f"trait_input={args.input}")
    print(f"trait_output={output_path}")
    print(f"rows={len(output)}")
    print(f"missing_yield={int(output['Yield_Mg_ha'].isna().sum())}")


def summarize_csv(args):
    df = pd.read_csv(args.input, nrows=0)
    rows = sum(1 for _ in open(args.input, "r", encoding=args.encoding)) - 1
    print(f"file={args.input}")
    print(f"rows={max(rows, 0)}")
    print(f"columns={len(df.columns)}")


def add_genotype_args(parser):
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-format", choices=["auto", "csv", "vcf"], default="auto")
    parser.add_argument("--id-col", default="Hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-value", type=float, default=-1)
    parser.add_argument("--low-variance-threshold", type=float, default=0.95)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--skip-imputation", action="store_true")
    parser.add_argument("--skip-low-variance", action="store_true")
    parser.add_argument("--skip-similarity-pruning", action="store_true")


def add_weather_args(parser):
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-col", default="Env")
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--missing", choices=["drop", "median", "zero"], default="drop")


def add_trait_args(parser):
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-col", default="Env")
    parser.add_argument("--hybrid-col", default="Hybrid")
    parser.add_argument("--yield-col", default="Yield_Mg_ha")
    parser.add_argument("--allow-missing-yield", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description="Preprocess genotype, weather, and trait files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    genotype = subparsers.add_parser("genotype", help="Clean genotype markers.")
    add_genotype_args(genotype)

    weather = subparsers.add_parser("weather", help="Keep manuscript weather variables.")
    add_weather_args(weather)

    trait = subparsers.add_parser("trait", help="Keep Env, Hybrid, and Yield_Mg_ha columns.")
    add_trait_args(trait)

    summary = subparsers.add_parser("summary", help="Count CSV rows and columns.")
    summary.add_argument("--input", required=True)
    summary.add_argument("--encoding", default="utf-8")

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "genotype":
        preprocess_genotype(args)
    elif args.command == "weather":
        clean_weather(args)
    elif args.command == "trait":
        clean_trait(args)
    elif args.command == "summary":
        summarize_csv(args)


if __name__ == "__main__":
    main()
