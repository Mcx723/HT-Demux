#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import gzip
import pandas as pd
import numpy as np
from scipy.io import mmread
import anndata
import scanpy as sc
import scanpy.external as sce


input_dir = "./data/HashSolo"
output_dir = "./results/HashSolo"
os.makedirs(output_dir, exist_ok=True)

script_start_time = time.time()


def pick_file(path_plain):
    path_gz = path_plain + ".gz"
    if os.path.exists(path_gz):
        return path_gz
    if os.path.exists(path_plain):
        return path_plain
    raise FileNotFoundError(f"Cannot find {path_plain} or {path_gz}")


def open_text_auto(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def make_unique(names):
    seen = {}
    out = []
    for name in names:
        name = str(name)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


# =========================
# Read input
# =========================
matrix_path = pick_file(os.path.join(input_dir, "matrix.mtx"))
barcodes_path = pick_file(os.path.join(input_dir, "barcodes.tsv"))
features_path = pick_file(os.path.join(input_dir, "features.tsv"))

# matrix.mtx is HTO x droplets, AnnData should be droplets x HTO
matrix_sparse = mmread(matrix_path).T.tocsr()

with open_text_auto(barcodes_path) as fh:
    barcodes = [line.strip().split("\t")[0] for line in fh if line.strip()]

features_df = pd.read_csv(
    features_path,
    sep="\t",
    header=None,
    dtype=str,
    compression="gzip" if features_path.endswith(".gz") else None
)

features = features_df.iloc[:, 0].astype(str).tolist()
features = make_unique(features)

adata = anndata.AnnData(X=matrix_sparse)
adata.obs_names = barcodes
adata.var_names = features

print(f"Input matrix dimension: {adata.n_vars} HTOs x {adata.n_obs} droplets")
print(f"scanpy version: {sc.__version__}")


# =========================
# Add HTO counts to obs
# =========================
hto_counts_df = pd.DataFrame(
    adata.X.toarray(),
    index=adata.obs_names,
    columns=adata.var_names
)

adata.obs = pd.concat([adata.obs, hto_counts_df], axis=1)


# =========================
# Run HashSolo with runtime
# =========================
method_start_time = time.time()

sce.pp.hashsolo(
    adata,
    cell_hashing_columns=features,
    inplace=True
)

method_end_time = time.time()
method_runtime_sec = method_end_time - method_start_time

print(f"HashSolo method runtime: {method_runtime_sec:.4f} seconds")


# =========================
# Save classification
# =========================
classification_df = adata.obs.copy()
classification_df.to_csv(
    os.path.join(output_dir, "HashSolo_classification.csv")
)


# =========================
# Save summary
# =========================
total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
top_indices = np.asarray(adata.X.argmax(axis=1)).ravel()

classification_col = "classification"
if classification_col not in adata.obs.columns:
    possible_cols = [c for c in adata.obs.columns if "class" in c.lower()]
    classification_col = possible_cols[0] if len(possible_cols) > 0 else None

summary_df = pd.DataFrame({
    "droplet_id": adata.obs_names,
    "nHTO_total": total_counts,
    "HTO_maxID": [features[int(i)] for i in top_indices],
    "final_assign": adata.obs[classification_col].values if classification_col is not None else pd.NA
})

summary_df.to_csv(
    os.path.join(output_dir, "HashSolo_summary.csv"),
    index=False
)


# =========================
# Save runtime
# =========================
script_end_time = time.time()
total_runtime_sec = script_end_time - script_start_time

runtime_df = pd.DataFrame([{
    "method": "HashSolo",
    "input_dir": input_dir,
    "n_hto": adata.n_vars,
    "n_droplets": adata.n_obs,
    "method_runtime_sec": method_runtime_sec,
    "total_runtime_sec": total_runtime_sec,
    "scanpy_version": sc.__version__,
    "classification_col": classification_col
}])

runtime_df.to_csv(
    os.path.join(output_dir, "HashSolo_runtime.csv"),
    index=False
)

print(f"HashSolo total runtime: {total_runtime_sec:.4f} seconds")
print("Classification column:", classification_col)
print("Finished HashSolo.")