import torch
import numpy as np
import pandas as pd
import itertools
import gzip
from pathlib import Path
from scipy.io import mmread
from typing import List, Optional, Dict, Tuple
from models.base_cluster import GMMCluster, NBCluster
from models.paramshared_cluster import ParamSharedCluster
from models.optimizer import fit_em, fit_gd

class DemuxManager:
    """Handles data I/O and orchestrates the demultiplexing workflow."""
    def __init__(self):
        self.tag_names: List[str] = []
        self.configs: Optional[torch.Tensor] = None
        self.cfg_labels: List[str] = []

    def load_10x_mtx(self, mtx_path: Path, features_path: Path, barcodes_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
        features = pd.read_csv(features_path, sep="\t", header=None, dtype=str)
        barcodes = pd.read_csv(barcodes_path, header=None, dtype=str)[0].values
        
        # Detect HTO/Antibody tags
        mask = features.iloc[:, 2].str.lower().str.contains("hto|antibody", na=False) if features.shape[1] >= 3 else np.ones(len(features), dtype=bool)
        row_idx = features[mask].index.to_numpy()
        self.tag_names = features.iloc[row_idx, 1].tolist()
        
        # Load Sparse Matrix
        open_func = gzip.open if mtx_path.suffix == ".gz" else open
        with open_func(mtx_path, "rb") as fh:
            sparse_data = mmread(fh).tocsr()
            counts_np = sparse_data[row_idx, :].toarray().T
        
        keep = counts_np.sum(axis=1) > 0
        return torch.tensor(counts_np[keep], dtype=torch.float32), barcodes[keep]

    def build_configs(self, max_klet: int = 2) -> torch.Tensor:
        H = len(self.tag_names)
        cfgs, labels = [np.zeros(H)], ["Negative"]
        for r in range(1, max_klet + 1):
            for comb in itertools.combinations(range(H), r):
                v = np.zeros(H)
                v[list(comb)] = 1.0
                cfgs.append(v)
                labels.append("+".join(self.tag_names[i] for i in comb))
        
        self.configs = torch.tensor(np.stack(cfgs), dtype=torch.float32)
        self.cfg_labels = labels
        return self.configs

    def run_demux(self, counts: torch.Tensor, model_type: str = "NB", method: str = "EM", device: str = "cpu", **kwargs) -> Dict:
        # Resolve config before model creation to avoid Type Errors
        configs = self.configs if self.configs is not None else self.build_configs()
        counts = counts.to(device)
        H = len(self.tag_names)
        
        if model_type.upper() == "GMM":
            log_c = torch.log1p(counts)
            x_in, c_in = log_c - log_c.mean(dim=1, keepdim=True), None
            base = GMMCluster(H)
        else:
            x_in, c_in = counts, counts
            base = NBCluster(H)

        model = ParamSharedCluster(base, configs).to(device)
        fit_func = fit_em if method.upper() == "EM" else fit_gd
        
        labels = fit_func(model, X=x_in, counts=c_in, **kwargs)

        return {
            "posteriors": model.posteriors,
            "cfg_labels": self.cfg_labels,
            "tag_names": self.tag_names
        }