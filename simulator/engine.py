import torch
import numpy as np
from abc import ABC, abstractmethod

class CellEngine(ABC):
    """细胞引擎基类：负责潜在状态分配与 Log 空间采样"""
    def __init__(self, registry):
        self.registry = registry

    def _sample_latent_space(self, n_cells):
        """核心逻辑：生成基于 GMM 逻辑的潜在对数矩阵"""
        h = self.registry.h_tags
        donor_indices = np.random.choice(h, size=n_cells)
        amb = torch.from_numpy(self.registry.ambient_vector).float()
        
        latent_matrix = torch.zeros((n_cells, h))
        
        for i, hto in enumerate(self.registry.htos):
            p = self.registry.profiles[hto]
            
            # 1. 采样背景噪声分布并融入全局背景
            bg_noise = torch.normal(p['mu_bg'], p['sigma_bg'], size=(n_cells,))
            latent_matrix[:, i] = bg_noise + amb[i]
            
            # 2. 采样供体阳性信号分布并融入全局背景
            is_donor = (donor_indices == i)
            sig_noise = torch.normal(p['mu_sig'], p['sigma_sig'], size=(int(is_donor.sum()),))
            latent_matrix[is_donor, i] = sig_noise + amb[i]
            
        return latent_matrix, donor_indices

    @abstractmethod
    def generate(self, n_cells):
        pass


class GMMCellEngine(CellEngine):
    """连续空间生成器：输出模拟 CLR/Log 后的数据"""
    def generate(self, n_cells):
        latent_matrix, donor_indices = self._sample_latent_space(n_cells)
        return latent_matrix, donor_indices


class NBCellEngine(CellEngine):
    """离散空间生成器：输出模拟原始 Counts 的数据"""
    def generate(self, n_cells):
        latent_matrix, donor_indices = self._sample_latent_space(n_cells)
        # 通过 Poisson 过程将连续密度投影为离散计数
        counts = torch.poisson(torch.exp(latent_matrix))
        return counts.int(), donor_indices