# SKILLCRAFT TASK 02 MACHINE LEARNING

## Problem statement
Group retail customers into meaningful segments using unsupervised clustering so the business can target marketing, offers, and retention strategies. We use the provided `Mall_Customers.csv` which contains customer demographics and two core behavioral features: `Annual Income (k$)` and `Spending Score (1-100)`.

## What I built
- `SCT_ML_02.py`: a high-level Python script that runs K-Means clustering on `Mall_Customers.csv`.
  - Default behavior: uses `Annual Income (k$)` and `Spending Score (1-100)`.
  - Prints cluster counts, cluster centroids (mean income and spending score), and the first 20 assignments.
  - Optional: save cluster assignments to CSV with `--save`.

## Approach (high level)
1. Load `Mall_Customers.csv`.
2. Select numeric features relevant to purchase behavior: `Annual Income (k$)` and `Spending Score (1-100)`.
3. Standardize features with `StandardScaler` to make clustering insensitive to scale.
4. Run `KMeans` with a user-specified number of clusters `k` (default 5).
5. Present cluster counts and centroid values (converted back into original units) so the business can interpret each segment.

## How to run
From PowerShell in the project folder:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas scikit-learn numpy
python SCT_ML_02.py --csv Mall_Customers.csv --k 5 --save --out-prefix mall_kmeans
```

- `--save` will write `{out-prefix}_clusters.csv` containing the original rows plus a `Cluster` column.
- If you only want to print results without saving, omit `--save`.

## Example output (what to expect)
- Cluster counts: a dictionary mapping cluster id -> number of customers.
- Cluster centroids: table showing mean `Annual Income (k$)` and mean `Spending Score (1-100)` per cluster.
- First 20 assignments: a small sample of rows with assigned cluster labels.

## Libraries used
- `pandas` and `numpy` for data handling.
- `scikit-learn` for `StandardScaler` and `KMeans`.

## Next steps / suggestions
- Automatically choose `k` using elbow or silhouette methods and show recommended values.
- Add more features for richer segmentation (age, gender, derived recency/frequency/monetary features if available).
- Persist the scaler and model for use in production scoring of new customers.
- Visualize clusters in 2D with labels and centroids for stakeholder presentations.

If you want, I can run the script here and show the live output, or save a CSV summary of clusters. Let me know which you prefer.
