import torch
import numpy as np
import pandas as pd
import itertools
from math import comb as n_choose_k
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

    def build_configs(self, max_klet: Optional[int] = None,
                      max_auto_configs: int = 4096,
                      configuration_prior: str = 'enumerated') -> torch.Tensor:
        H = len(self.tag_names)
        if H < 1 or max_auto_configs < 1:
            raise ValueError('Tags and a positive higher-order configuration budget are required')
        requested = max_klet
        if max_klet is None:
            # Always retain singles/pairs. Add higher orders only when their
            # complete configuration space fits a fixed, GT-independent budget.
            max_klet = min(2, H)
            for order in range(3, min(4, H) + 1):
                if sum(n_choose_k(H, k) for k in range(order + 1)) > max_auto_configs:
                    break
                max_klet = order
        elif not isinstance(max_klet, int) or max_klet < 1:
            raise ValueError('max_klet must be a positive integer or None (auto)')
        max_klet = min(max_klet, H)
        self.config_policy = dict(requested_max_klet=requested, resolved_max_klet=max_klet,
                                  max_auto_configs=max_auto_configs,
                                  n_configs=sum(n_choose_k(H, k) for k in range(max_klet + 1)))
        if configuration_prior == 'factorized' or (
                configuration_prior == 'auto' and requested is None and max_klet < min(4, H)):
            if requested is not None:
                raise ValueError('Implicit combinations cannot be combined with an explicit max_klet')
            self.config_policy['n_configs_enumerated_reference'] = self.config_policy['n_configs']
            self.config_policy['n_configs'] = 0
            self.configs = torch.empty((0, H), dtype=torch.float32)
            self.cfg_labels = []
            return self.configs
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

    def run_demux(self, counts: torch.Tensor, model_type: str = "NB", method: str = "EM", device: str = "cpu", configuration_prior: str = "enumerated", signal_shape_min: float = 1.0, init_tail_fraction: Optional[float] = None, **kwargs) -> Dict:
        # Resolve config before model creation to avoid Type Errors
        configs = self.configs if self.configs is not None else self.build_configs()
        counts = counts.to(device)
        H = len(self.tag_names)
        requested_type = model_type.upper()
        if requested_type == 'AUTO':
            model_type = 'NB'
        
        if model_type.upper() == "GMM":
            log_c = torch.log1p(counts)
            x_in, c_in = log_c - log_c.mean(dim=1, keepdim=True), None
            base = GMMCluster(H)
        elif model_type.upper() == 'DLN':
            from models.base_cluster import DiscreteLogNormalCluster
            x_in, c_in = counts, counts
            base = DiscreteLogNormalCluster(H)
        else:
            x_in, c_in = counts, counts
            base = NBCluster(H, signal_shape_min=signal_shape_min,
                             init_tail_fraction=init_tail_fraction)

        if configuration_prior not in ['auto', 'enumerated', 'factorized']:
            raise ValueError('Unknown configuration prior')
        policy = getattr(self, 'config_policy', {})
        implicit = model_type.upper() == 'DLN' or configuration_prior == 'factorized' or (
            configuration_prior == 'auto' and model_type.upper() == 'NB'
            and method.upper() == 'EM' and 'requested_max_klet' in policy
            and policy.get('requested_max_klet') is None
            and policy.get('resolved_max_klet', H) < min(4, H))
        if implicit:
            if model_type.upper() not in ['NB', 'DLN'] or method.upper() != 'EM' or policy.get('requested_max_klet') is not None:
                raise ValueError('Factorized prior requires count soft EM and no explicit max_klet')
            from models.optimizer import FactorizedSharedCluster, fit_factorized_em
            families = ['NB', 'DLN'] if requested_type == 'AUTO' else [model_type.upper()]
            candidates = []
            for family in families:
                if family == 'DLN':
                    from models.base_cluster import DiscreteLogNormalCluster
                    candidate_base = DiscreteLogNormalCluster(H)
                else:
                    candidate_base = NBCluster(H, signal_shape_min=signal_shape_min,
                                               init_tail_fraction=init_tail_fraction)
                candidate = FactorizedSharedCluster(candidate_base).to(device)
                fit_factorized_em(candidate, X=x_in, counts=c_in, **kwargs)
                n_parameters = 5 * H + 1
                bic = -2 * len(counts) * candidate.fit_diagnostics['final_mean_log_likelihood'] + n_parameters * np.log(len(counts))
                candidates.append((bic, family, candidate))
            _, selected_family, model = min(candidates, key=lambda item: item[0])
            base = model.base_cluster
            model.fit_diagnostics['emission_selection'] = {
                'criterion': 'BIC on normalized integer-count likelihood; no GT',
                'selected_family': selected_family,
                'candidates': [{'family': f, 'bic': b, 'n_parameters': 5 * H + 1,
                                'mean_log_likelihood': m.fit_diagnostics['final_mean_log_likelihood']}
                               for b, f, m in candidates]}
            self.config_policy = {**policy, 'resolved_max_klet': H,
                                  'n_configs': 0, 'n_configs_represented': str(2 ** H),
                                  'inference': 'exact_implicit_all_tag_sets'}
            model.fit_diagnostics['configuration_policy'] = self.config_policy
            return {'posteriors': None, 'factorized_model': model,
                    'cfg_labels': self.cfg_labels, 'tag_names': self.tag_names,
                    'fit_diagnostics': model.fit_diagnostics,
                    'fitted_parameters': {name: value.detach().cpu().tolist() for name, value in base.named_parameters()}}

        if len(configs) == 0:
            raise ValueError('Materialize configurations or request implicit inference')
        model = ParamSharedCluster(base, configs).to(device)
        fit_func = fit_em if method.upper() == "EM" else fit_gd
        
        labels = fit_func(model, X=x_in, counts=c_in, **kwargs)
        model.fit_diagnostics = getattr(model, 'fit_diagnostics', {})
        model.fit_diagnostics['configuration_policy'] = getattr(
            self, 'config_policy', {'requested_max_klet': 'external_configs',
                                   'resolved_max_klet': int(configs.sum(1).max().item()),
                                   'n_configs': len(configs)})

        return {
            "posteriors": model.posteriors,
            "cfg_labels": self.cfg_labels,
            "tag_names": self.tag_names,
            "fit_diagnostics": getattr(model, "fit_diagnostics", {}),
            "fitted_parameters": {name: value.detach().cpu().tolist() for name,value in base.named_parameters()},
        }
