# data.py
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import itertools
import torch
from scipy.io import mmread
from typing import Tuple, List, Optional


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def load_mtx(
    features_path: Path,
    barcodes_path: Path,
    mtx_path: Path
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Read 10x-style matrix + features + barcodes and return HTO subset.

    Returns:
      - counts: torch.Tensor shape (N_droplets, H) float32 (filtered to non-zero droplets)
      - barcodes_kept: list[str] length N_droplets
      - tag_names: list[str] length H
    """
    features = read_table(features_path, sep="\t", header=None, dtype=str)
    barcodes = read_table(barcodes_path, header=None, dtype=str)[0].to_numpy()

    # load sparse matrix (support .gz)
    if mtx_path.suffix == ".gz":
        with gzip.open(mtx_path, "rt") as fh:
            sparse_mtx = mmread(fh).tocsr()
    else:
        sparse_mtx = mmread(str(mtx_path)).tocsr()

    # detect HTO rows (3rd column often contains "antibody" or "hto")
    if features.shape[1] >= 3:
        type_col = features.iloc[:, 2].str.lower()
        mask = type_col.str.contains("hto|antibody|antibody capture", na=False)
        if mask.sum() > 0:
            hto_df = features[mask]
            tag_names = hto_df.iloc[:, 1].tolist()
            row_idx = hto_df.index.to_numpy()
        else:
            tag_names = features.iloc[:, 1].tolist()
            row_idx = np.arange(len(features))
    else:
        tag_names = features.iloc[:, 1].tolist()
        row_idx = np.arange(len(features))

    # extract and transpose -> (cells, H)
    sub_arr = sparse_mtx[row_idx, :].toarray().T  # (cells, H)
    keep_mask = (sub_arr.sum(axis=1) > 0)
    sub_kept = sub_arr[keep_mask]
    barcodes_kept = barcodes[keep_mask].tolist()

    counts = torch.tensor(sub_kept, dtype=torch.float32)
    return counts, barcodes_kept, tag_names


def gen_configs(H: int, klet: int, tag_names: Optional[List[str]] = None
               ) -> Tuple[torch.Tensor, List[str]]:
    """
    Enumerate configs (0/1 vectors) with up to klet positives.

    Returns:
      - configs: torch.Tensor shape (K, H) float32
      - cfg_labels: list[str] human-readable labels (uses tag_names if provided)
    Order: popcount increasing (none -> singlets -> doublets -> ...), combos in lexicographic order.
    """
    if H <= 0:
        raise ValueError("No tag recognized.")
    klet = min(klet, H)

    cfgs = [np.zeros(H, dtype=np.uint8)]
    labels = ["none"]

    for r in range(1, klet + 1):
        for comb in itertools.combinations(range(H), r):
            v = np.zeros(H, dtype=np.uint8)
            for i in comb:
                v[i] = 1
            cfgs.append(v)
            if tag_names is None:
                labels.append("+".join(f"S{i}" for i in comb))
            else:
                labels.append("+".join(tag_names[i] for i in comb))

    cfg_arr = np.stack(cfgs, axis=0)
    return torch.tensor(cfg_arr, dtype=torch.float32), labels
