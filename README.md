# K-Means Clustering for Retail Customers

## Overview
This small project provides a reusable Python script to cluster retail customers based on their purchase history using K-Means. It loads a CSV (defaults to `Mall_Customers.csv`), selects numeric purchase-related features, preprocesses and scales them, suggests an appropriate number of clusters using elbow and silhouette diagnostics, fits K-Means, and writes cluster assignments to a CSV.

## Files
- `kmeans_clustering.py`: Main script to run clustering and diagnostics.
- `requirements.txt`: Python dependencies.

## Installation
Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
Basic usage (defaults expect `Mall_Customers.csv` in the same folder):

```powershell
python kmeans_clustering.py --plot --save-model
```

Specify a CSV and columns to use (comma-separated):

```powershell
python kmeans_clustering.py --csv data/customers.csv --columns "Annual Income (k$),Spending Score (1-100)" --plot
```

Force a specific `k`:

```powershell
python kmeans_clustering.py --k 5 --csv Mall_Customers.csv
```

Outputs:
- `kmeans_output_clusters.csv`: original rows with an added `Cluster` column (prefix can be changed with `--out-prefix`).
- `kmeans_output_elbow_silhouette.png` (if `--plot` used): diagnostic plot showing inertia and silhouette scores across k.
- `kmeans_output_kmeans.joblib` (if `--save-model` used): saved model + scaler + selected feature list.

## What we used and why
- Data loading: `pandas` for CSV reading and easy dataframe manipulation.
- Numeric feature selection: we automatically select numeric columns (dropping ID-like fields). You can override with `--columns`.
- Missing values: rows with missing values in selected features are dropped (simple, transparent approach). For production you'd consider imputation.
- Scaling: `StandardScaler` from `scikit-learn` to standardize features before K-Means (important because K-Means is distance-based).
- K-Means: `sklearn.cluster.KMeans` for clustering. `n_init='auto'` lets scikit-learn choose a sensible default for stability.
- Diagnostics:
  - Elbow (inertia): shows how within-cluster sum-of-squares decreases with k.
  - Silhouette score: measures how well-separated clusters are. We prefer silhouette when it's computable.
- Persistence: `joblib` to save fitted model and scaler for later use.

## How to choose k (short)
- Run without `--k` and with `--plot`. Examine the elbow plot for the point where inertia decrease slows.
- Check silhouette scores: the k with the highest silhouette is often a good choice.
- Use domain knowledge (business meaning of clusters) to finalize the choice.

## Next steps / Improvements
- Add robust missing-value handling (imputation), categorical feature encoding, and feature engineering from transaction histories.
- Persist preprocessing pipeline (e.g., `sklearn.pipeline`) so you can transform new customers consistently.
- Add unit tests, a small sample dataset, and a notebook for exploratory analysis and interactive tuning.

---
If you want, I can also:
- Create an example Jupyter notebook demonstrating clustering on your `Mall_Customers.csv`.
- Add imputation or encode categorical data.
- Commit these files to git and run a quick test run on the CSV you have.
