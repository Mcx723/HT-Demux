import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_cluster import BaseCluster

class ParamSharedCluster(nn.Module):
    """The backbone framework relating tags to experiment design."""
    def __init__(self, base_cluster: BaseCluster, configs: torch.Tensor):
        super().__init__()
        self.base_cluster = base_cluster
        # 使用 register_buffer 确保 configs 随模型移动 (CPU/GPU)
        self.register_buffer('configs', configs.float())
        self.K, self.H = configs.shape
        self.pi_logits = nn.Parameter(torch.zeros(self.K))
        self.posteriors: Optional[torch.Tensor] = None

    @property
    def log_pi(self) -> torch.Tensor:
        return F.log_softmax(self.pi_logits, dim=0)

    def calculate_posterior(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the log-posterior probability following the successful NumPy logic.
        P(Z|X) ∝ P(X|Z)P(Z)
        """
        # 1. 获取每个通道的 log-likelihood (N, H)
        ll = self.base_cluster.compute_probability(X, counts)
        log_sig = ll['sig']
        log_bg = ll['bg']

        # 2. 核心逻辑：全背景分 + 信号/背景的差分
        # base_bg_total: 每个 cell 假设所有 HTO 都是背景时的总分 (N, 1)
        base_bg_total = log_bg.sum(dim=1, keepdim=True)
        
        # diff: 信号比背景多出的分数 (N, H)
        diff = log_sig - log_bg
        
        # 3. 组合成 K 个配置的似然分数 (N, K)
        # configs.T 形状 (H, K), 结果为 (N, K)
        log_prob_x_z = base_bg_total + diff @ self.configs.T
        
        # 4. 加入先验分布 pi (Mixing proportions)
        log_joint = log_prob_x_z + self.log_pi.unsqueeze(0)
        
        # 5. 计算后验概率用于 M-step 和分类
        self.posteriors = F.softmax(log_joint, dim=1)
        
        return log_joint