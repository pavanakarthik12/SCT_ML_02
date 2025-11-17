import argparse
import logging
import os
from typing import List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded data with shape {df.shape} from {csv_path}")
    return df


def select_numeric_features(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in CSV: {missing}")
        selected = df[columns]
    else:
        # choose numeric columns and drop obvious ID columns
        numeric = df.select_dtypes(include=[np.number]).copy()
        id_like = [c for c in numeric.columns if 'id' in c.lower() or c.lower().startswith('customer')]
        selected = numeric.drop(columns=id_like, errors='ignore')
    if selected.shape[1] == 0:
        raise ValueError("No numeric features found for clustering. Provide columns explicitly.")
    logging.info(f"Selected features: {list(selected.columns)}")
    return selected


def preprocess(X: pd.DataFrame, scale: bool = True) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    X_proc = X.copy()
    # simple missing value handling: drop rows with any NA
    before = X_proc.shape[0]
    X_proc = X_proc.dropna()
    after = X_proc.shape[0]
    if after < before:
        logging.info(f"Dropped {before - after} rows containing NA values.")
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_proc.values)
    else:
        X_arr = X_proc.values
    return X_arr, scaler


def find_optimal_k(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[List[float], List[float]]:
    inertias = []
    silhouettes = []
    Ks = list(range(k_min, k_max + 1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # silhouette requires at least 2 clusters and less than n_samples
        if 1 < k < X.shape[0]:
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = float('nan')
        else:
            score = float('nan')
        silhouettes.append(score)
        logging.info(f"k={k}: inertia={km.inertia_:.2f}, silhouette={score:.4f}")
    return inertias, silhouettes


def plot_elbow_silhouette(Ks: List[int], inertias: List[float], silhouettes: List[float], out_prefix: str):
    sns.set(style='whitegrid')
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(Ks, inertias, '-o', color='C0')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia (Sum of squared distances)', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(Ks, silhouettes, '-s', color='C1')
    ax2.set_ylabel('Silhouette Score', color='C1')
    plt.title('Elbow (inertia) and Silhouette by k')
    plt.tight_layout()
    fname = f"{out_prefix}_elbow_silhouette.png"
    fig.savefig(fname)
    plt.close(fig)
    logging.info(f"Saved elbow + silhouette plot to {fname}")


def run_kmeans(X: np.ndarray, n_clusters: int) -> Tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    logging.info(f"Fitted KMeans with k={n_clusters}")
    return km, labels


def save_outputs(original_df: pd.DataFrame, indices: pd.Index, labels: np.ndarray, out_csv: str):
    # original_df was possibly reduced by dropping NA rows; indices maps back to those rows
    res = original_df.loc[indices].copy()
    res['Cluster'] = labels
    res.to_csv(out_csv, index=False)
    logging.info(f"Saved cluster assignments to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description='K-Means clustering for retail customers (purchase history)')
    p.add_argument('--csv', type=str, default='Mall_Customers.csv', help='Path to input CSV file')
    p.add_argument('--columns', type=str, default=None, help='Comma-separated list of columns to use (default: auto numeric selection)')
    p.add_argument('--k', type=int, default=None, help='Number of clusters to fit (if not set, use elbow/silhouette to suggest)')
    p.add_argument('--max-k', type=int, default=10, help='Maximum k to try when searching for optimal k')
    p.add_argument('--no-scale', dest='scale', action='store_false', help='Disable feature scaling')
    p.add_argument('--plot', action='store_true', help='Generate and save diagnostic plots')
    p.add_argument('--out-prefix', type=str, default='kmeans_output', help='Prefix for output files')
    p.add_argument('--save-model', action='store_true', help='Save fitted KMeans model with joblib')
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(args.csv)
    cols = args.columns.split(',') if args.columns else None
    features_df = select_numeric_features(df, cols)

    # Keep index mapping for rows after dropna
    features_df_before = features_df.copy()
    X_arr, scaler = preprocess(features_df, scale=args.scale)
    # compute the indices that remain after dropping NA in preprocess
    indices = features_df_before.dropna().index

    if args.k is None:
        k_min = 2
        k_max = max(2, args.max_k)
        inertias, silhouettes = find_optimal_k(X_arr, k_min=k_min, k_max=k_max)
        Ks = list(range(k_min, k_max + 1))
        # Choose k by silhouette if available, otherwise elbow heuristic (largest drop in inertia)
        best_sil_idx = int(np.nanargmax(silhouettes)) if not all(np.isnan(silhouettes)) else None
        if best_sil_idx is not None and not np.isnan(silhouettes[best_sil_idx]):
            suggested_k = Ks[best_sil_idx]
        else:
            # elbow: largest relative drop in inertia
            rel_changes = np.diff(inertias) / inertias[:-1]
            suggested_k = Ks[int(np.argmin(rel_changes)) + 1] if len(rel_changes) > 0 else Ks[0]
        logging.info(f"Suggested k={suggested_k}")
        chosen_k = suggested_k
        if args.plot:
            plot_elbow_silhouette(Ks, inertias, silhouettes, args.out_prefix)
    else:
        chosen_k = args.k

    km, labels = run_kmeans(X_arr, n_clusters=chosen_k)
    out_csv = f"{args.out_prefix}_clusters.csv"
    save_outputs(df, indices, labels, out_csv)

    if args.save_model:
        model_path = f"{args.out_prefix}_kmeans.joblib"
        joblib.dump({'model': km, 'scaler': scaler, 'features': list(features_df.columns)}, model_path)
        logging.info(f"Saved model + scaler to {model_path}")

    # quick cluster summary
    unique, counts = np.unique(labels, return_counts=True)
    summary = dict(zip(unique.tolist(), counts.tolist()))
    logging.info(f"Cluster counts: {summary}")


if __name__ == '__main__':
    main()
