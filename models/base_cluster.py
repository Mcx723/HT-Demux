import torch
import torch.nn as nn
from typing import Optional, Dict

EPS = 1e-12

class BaseCluster(nn.Module):
    """Abstract base for mixture distributions."""
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def init_parameters(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def compute_probability(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class GMMCluster(BaseCluster):
    """Gaussian distribution for CLR-normalized data."""
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.mu_sig = nn.Parameter(torch.ones(num_features))
        self.mu_bg = nn.Parameter(torch.zeros(num_features))
        # 使用 log_sigma 确保标准差永远为正
        self.log_sigma_sig = nn.Parameter(torch.zeros(num_features))
        self.log_sigma_bg = nn.Parameter(torch.zeros(num_features))

    def compute_probability(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        def log_gaussian(x, mu, log_sigma):
            # 对齐标准高斯公式：-0.5 * log(2*pi*sigma^2) - (x-mu)^2 / (2*sigma^2)
            sigma = torch.exp(log_sigma)
            var = sigma ** 2 + EPS
            return -0.5 * ((x - mu)**2 / var) - log_sigma - 0.5 * torch.log(torch.tensor(2 * torch.pi))

        ll_sig = log_gaussian(X, self.mu_sig.unsqueeze(0), self.log_sigma_sig.unsqueeze(0))
        ll_bg = log_gaussian(X, self.mu_bg.unsqueeze(0), self.log_sigma_bg.unsqueeze(0))
        return {'sig': ll_sig, 'bg': ll_bg}

    def init_parameters(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None):
        """严格对齐高标准初始化逻辑：使用更具代表性的分位数划分"""
        with torch.no_grad():
            N, H = X.shape
            for h in range(H):
                col = X[:, h]
                # 模拟数据中信号通常位于 95% 分位以上
                q_high = torch.quantile(col, 0.95)
                q_low = torch.quantile(col, 0.50)
                
                high_mask = col > q_high
                low_mask = col <= q_low
                
                # 初始化均值
                m_sig = col[high_mask].mean()
                m_bg = col[low_mask].mean()
                
                self.mu_sig[h] = m_sig
                self.mu_bg[h] = m_bg
                
                # 初始化标准差：使用各自组内的 std，并设置合理的最小值防止塌陷
                s_sig = col[high_mask].std() if high_mask.sum() > 1 else X[:, h].std()
                s_bg = col[low_mask].std() if low_mask.sum() > 1 else X[:, h].std()
                
                self.log_sigma_sig[h] = torch.log(torch.clamp(s_sig, min=0.1))
                self.log_sigma_bg[h] = torch.log(torch.clamp(s_bg, min=0.1))

class NBCluster(BaseCluster):
    """Negative Binomial distribution for raw counts."""
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.mu_sig = nn.Parameter(torch.ones(num_features))
        self.mu_bg = nn.Parameter(torch.ones(num_features) * 0.1)
        self.log_r_sig = nn.Parameter(torch.ones(num_features))
        self.log_r_bg = nn.Parameter(torch.ones(num_features))

    def compute_probability(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        r_s, r_b = torch.exp(self.log_r_sig), torch.exp(self.log_r_bg)
        m_s, m_b = torch.clamp(self.mu_sig, min=EPS), torch.clamp(self.mu_bg, min=EPS)
        
        def log_nb(k, mu, r):
            kd = k.double()
            rd = r.unsqueeze(0).double()
            mud = mu.unsqueeze(0).double()
            
            ll = torch.lgamma(kd + rd) - torch.lgamma(kd + 1.0) - torch.lgamma(rd) + \
                 rd * (torch.log(rd + EPS) - torch.log(rd + mud + EPS)) + \
                 kd * (torch.log(mud + EPS) - torch.log(rd + mud + EPS))
            return ll.float()

        return {'sig': log_nb(counts, m_s, r_s), 'bg': log_nb(counts, m_b, r_b)}

    def init_parameters(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if counts is None: return
            N, H = counts.shape
            for h in range(H):
                col = counts[:, h].float()
                qv = torch.quantile(col, 0.9)
                low_mask = col <= qv
                high_mask = col > qv
                
                m_bg = col[low_mask].mean()
                m_sig = col[high_mask].mean()
                self.mu_bg[h] = torch.clamp(m_bg, min=1e-3)
                self.mu_sig[h] = torch.clamp(m_sig, min=m_bg * 2.0)
                
                def get_r(vals, mu):
                    var = vals.var()
                    r = (mu**2) / torch.clamp(var - mu, min=1.0)
                    return torch.log(torch.clamp(r, min=0.1, max=100.0))

                self.log_r_bg[h] = get_r(col[low_mask], self.mu_bg[h])
                self.log_r_sig[h] = get_r(col[high_mask], self.mu_sig[h])