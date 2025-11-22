"""
SCT_ML_02.py

High-level K-Means clustering for Mall Customers (retail customer segmentation).

This script groups customers using `Annual Income (k$)` and `Spending Score (1-100)`
from `Mall_Customers.csv` and prints cluster counts and centroids. Optionally
it can save assignments to CSV.

Usage (PowerShell):
    python SCT_ML_02.py --csv Mall_Customers.csv --k 5 --save --out-prefix mall_kmeans

"""
import argparse
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def select_mall_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Annual Income (k$)', 'Spending Score (1-100)']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for clustering: {missing}")
    return df[cols].copy()


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = km.fit_predict(Xs)
    centroids = scaler.inverse_transform(km.cluster_centers_)
    return labels, centroids


def parse_args():
    p = argparse.ArgumentParser(description='SCT_ML_02 - KMeans clustering for Mall_Customers.csv')
    p.add_argument('--csv', type=str, default='Mall_Customers.csv', help='Path to Mall_Customers.csv')
    p.add_argument('--k', type=int, default=5, help='Number of clusters')
    p.add_argument('--random-state', type=int, default=42, help='Random seed')
    p.add_argument('--save', action='store_true', help='Save cluster assignments to CSV')
    p.add_argument('--out-prefix', type=str, default='mall_kmeans', help='Prefix for output files')
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(args.csv)
    Xdf = select_mall_features(df)
    X = Xdf.values

    labels, centroids = run_kmeans(X, k=args.k, random_state=args.random_state)

    df_out = df.copy()
    df_out['Cluster'] = labels

    counts = df_out['Cluster'].value_counts().sort_index().to_dict()
    logging.info(f'Cluster counts: {counts}')

    cent_df = pd.DataFrame(centroids, columns=Xdf.columns)
    cent_df.index.name = 'Cluster'
    logging.info('Cluster centroids (means in original units):')
    logging.info('\n' + cent_df.to_string())

    print('\nSample assignments (first 20 rows):')
    print(df_out[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(20).to_string(index=False))

    if args.save:
        out_csv = f"{args.out_prefix}_clusters.csv"
        df_out.to_csv(out_csv, index=False)
        logging.info(f'Saved cluster assignments to {out_csv}')


if __name__ == '__main__':
    main()
