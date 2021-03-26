import argparse
import csv
import lzma
from pathlib import Path
from typing import IO

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


TOURNAMENT_NAME = "kazutsugi"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"


def feature_exposures(df):
    feature_names = [f for f in df.columns
                     if f.startswith("feature")]
    exposures = {}
    for f in feature_names:
        fe = spearmanr(df[PREDICTION_NAME], df[f])[0]
        exposures[f] = fe
    return pd.Series(exposures)


def neutralize(df, target=PREDICTION_NAME, by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series
    # is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] *
                                               len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @
                            (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()


def open_file(file_path: Path, mode: str = "rt") -> IO:
    if file_path.suffix == '.xz':
        return lzma.open(file_path, mode)
    else:
        return open(file_path, mode)


def read_csv(file_path):
    with open_file(file_path) as f:
        column_names = next(csv.reader(f))
    dtypes = {x: np.float32 for x in column_names if
              x.startswith(('feature', 'target'))}
    dtypes.update({x: np.float32 for x in column_names if
                   x.startswith(('prediction', 'probability'))})
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neutralize numerai predictions")
    parser.add_argument('tournament_data_file',
                        type=lambda p: Path(p).absolute())
    parser.add_argument('predictions_file', type=lambda p: Path(p).absolute())
    parser.add_argument('-p', '--proportion', type=float, default=1.0,
                        help="Neutralization proportion, defaults to 1.0")
    parser.add_argument('-o', '--output', type=Path,
                        default=Path.cwd() / "neutralized.csv",
                        help="Output filename, defaults to 'neutralized.csv'")
    parser.add_argument('-t', '--top-k', type=int,
                        default=None,
                        help="Neutralize only the top k features"
                             ", disabled by default")
    return parser.parse_args()


def rescale(df):
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    pred = df[PREDICTION_NAME].values.reshape(-1, 1)
    pred = pred.astype(np.float32)
    scaled = scaler.fit_transform(pred)
    scaled = scaled.reshape(scaled.shape[0])
    return pd.Series(scaled, df.index.values, name=PREDICTION_NAME)


def main():
    args = parse_args()
    tournament_data = read_csv(args.tournament_data_file)
    predictions = read_csv(args.predictions_file)
    merged = tournament_data.join(predictions)
    eras = tournament_data[["era"]]
    del tournament_data
    if args.top_k is not None:
        exposures = feature_exposures(merged)
        top = exposures.abs().nlargest(args.top_k)
        by = [x for x in top.index]
        print(f"Only neutralizing by the top {args.top_k} columns")
        print(" ".join(by))
    else:
        by = None
    neutralized = neutralize(merged, by=by, proportion=args.proportion)
    df = eras.join(neutralized).groupby("era", sort=False).apply(rescale)
    df = df.reset_index().drop("era", axis=1)
    df = df.set_index(df.columns[0])
    df.to_csv(args.output, index_label="id")


if __name__ == '__main__':
    main()