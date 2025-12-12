# model.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch

EPS = 1e-12


class BaseModel(ABC):
    """
    Abstract interface:
      - init_params(X, counts) -> dict[str, Tensor]
      - per_tag_loglik(X, counts, params) -> (N,H) Tensor (signal)
      - fit_params(X, counts, post, configs, params_old) -> dict[str, Tensor]
    """

    @abstractmethod
    def init_params(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def per_tag_loglik(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor],
                       params: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def fit_params(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor],
                   post: torch.Tensor, configs: torch.Tensor, params_old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    def _per_tag_loglik_bg(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor],
                           params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Optional background per-tag log-likelihood. Default raises NotImplementedError.
        Subclasses implementing background should override this.
        """
        raise NotImplementedError

    def loglik_model(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor],
                     params: Dict[str, torch.Tensor], configs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assemble per-cluster log-likelihoods from per-tag contributions.

        If params contains background params ('mu_bg' etc.), attempt to call subclass
        implementation of _per_tag_loglik_bg; if not implemented, fall back to signal loglik.
        """
        lg_sig = self.per_tag_loglik(X, counts, params)

        # compute lg_bg if available, else copy signal
        if 'mu_bg' in params:
            try:
                lg_bg = self._per_tag_loglik_bg(X, counts, params)
            except NotImplementedError:
                lg_bg = lg_sig.clone()
        else:
            lg_bg = lg_sig.clone()

        if configs is None:
            return lg_sig

        cfg = configs.to(dtype=torch.float32, device=lg_sig.device)  # (K,H)
        return lg_sig.matmul(cfg.t()) + lg_bg.matmul((1.0 - cfg).t())


# helpers
def _gmm_per_tag_loglik(X: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    mu_b = mu.view(1, -1)
    var_b = torch.clamp(var.view(1, -1), min=1e-12)
    diff = X - mu_b
    return -0.5 * (diff * diff) / var_b - 0.5 * torch.log(2.0 * torch.pi * var_b)


def _nb_logpmf_torch(k: torch.Tensor, mu: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    mu_b = mu.view(1, -1).to(dtype=torch.float64)
    r_b = r.view(1, -1).to(dtype=torch.float64)
    k_f = k.to(dtype=torch.float64)
    t1 = torch.lgamma(k_f + r_b) - torch.lgamma(k_f + 1.0) - torch.lgamma(r_b)
    t2 = r_b * (torch.log(r_b + EPS) - torch.log(r_b + mu_b + EPS))
    t3 = k_f * (torch.log(mu_b + EPS) - torch.log(r_b + mu_b + EPS))
    return (t1 + t2 + t3).to(dtype=torch.float32)


def _method_of_moments_r(col, eps=1e-6):
    import numpy as _np
    mu = float(_np.mean(col))
    var = float(_np.var(col))
    denom = max(var - mu, eps)
    return max((mu * mu) / denom, eps)


class GMMModel(BaseModel):
    def __init__(self, fallback_q: float = 0.9):
        self.fallback_q = fallback_q

    def init_params(self, X: Optional[torch.Tensor] = None, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if X is None:
            raise ValueError("GMMModel.init_params requires X (continuous input).")
        device = X.device
        N, H = X.shape
        import numpy as _np
        mu_bg = torch.zeros(H, device=device)
        mu_sig = torch.zeros(H, device=device)
        var_bg = torch.ones(H, device=device)
        var_sig = torch.ones(H, device=device)
        pi_sig = torch.zeros(H, device=device)

        Xn = X.detach().cpu().numpy()
        for h in range(H):
            col = Xn[:, h]
            qv = float(_np.quantile(col, self.fallback_q))
            low = col[col <= qv] if col.size > 0 else col
            high = col[col > qv] if col.size > 0 else col
            if low.size == 0:
                low = col
            if high.size == 0:
                high = col
            mu_bg[h] = float(low.mean())
            mu_sig[h] = float(high.mean())
            var_bg[h] = float(low.var() + 1e-8)
            var_sig[h] = float(high.var() + 1e-8)
            pi_sig[h] = float(max(1e-8, high.size / float(max(1, N))))
        return {'mu_bg': mu_bg, 'mu_sig': mu_sig, 'var_bg': var_bg, 'var_sig': var_sig, 'pi_sig': pi_sig}

    def per_tag_loglik(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if X is None:
            raise ValueError("GMMModel.per_tag_loglik requires X (continuous input).")
        return _gmm_per_tag_loglik(X, params['mu_sig'], params['var_sig'])

    def _per_tag_loglik_bg(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if X is None:
            raise ValueError("GMMModel._per_tag_loglik_bg requires X (continuous input).")
        return _gmm_per_tag_loglik(X, params['mu_bg'], params['var_bg'])

    def fit_params(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor], post: torch.Tensor,
                   configs: torch.Tensor, params_old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if X is None:
            raise ValueError("GMMModel.fit_params requires X (continuous input).")
        device = X.device
        configs = configs.to(device=device, dtype=torch.float32)
        q = post.to(device=device, dtype=torch.float32)

        S = q.matmul(configs)           # (N,H)
        B = q.matmul(1.0 - configs)

        sum_s = torch.clamp(S.sum(dim=0), min=EPS)
        sum_b = torch.clamp(B.sum(dim=0), min=EPS)

        sum_s_x = (S * X).sum(dim=0)
        sum_s_x2 = (S * (X * X)).sum(dim=0)
        sum_b_x = (B * X).sum(dim=0)
        sum_b_x2 = (B * (X * X)).sum(dim=0)

        mu_sig = sum_s_x / sum_s
        var_sig = (sum_s_x2 / sum_s) - (mu_sig * mu_sig)
        var_sig = torch.clamp(var_sig, min=1e-8)

        mu_bg = sum_b_x / sum_b
        var_bg = (sum_b_x2 / sum_b) - (mu_bg * mu_bg)
        var_bg = torch.clamp(var_bg, min=1e-8)

        pi_sig = sum_s / (sum_s + sum_b + EPS)

        return {'mu_bg': mu_bg, 'mu_sig': mu_sig, 'var_bg': var_bg, 'var_sig': var_sig, 'pi_sig': pi_sig}


class NBModel(BaseModel):
    def __init__(self, fallback_q: float = 0.9):
        self.fallback_q = fallback_q

    # accept Optional for static checking; runtime requires counts.
    def init_params(self, X: Optional[torch.Tensor] = None, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if counts is None:
            raise ValueError("NBModel.init_params requires counts (raw integer counts).")
        device = counts.device
        N, H = counts.shape
        import numpy as _np
        mu_bg = torch.zeros(H, device=device)
        mu_sig = torch.zeros(H, device=device)
        r_bg = torch.ones(H, device=device)
        r_sig = torch.ones(H, device=device)
        pi_sig = torch.zeros(H, device=device)

        cf = counts.detach().cpu().numpy()
        for h in range(H):
            col = cf[:, h].astype(float)
            qv = float(_np.quantile(col, self.fallback_q))
            low = col[col <= qv]
            high = col[col > qv]
            if low.size == 0:
                low = col
            if high.size == 0:
                high = col
            mu_bg[h] = float(low.mean())
            mu_sig[h] = float(high.mean())
            r_bg[h] = float(max(1e-6, _method_of_moments_r(low)))
            r_sig[h] = float(max(1e-6, _method_of_moments_r(high)))
            pi_sig[h] = float(max(1e-8, high.size / float(max(1, N))))

        mu_bg = torch.clamp(mu_bg, min=1e-8)
        mu_sig = torch.clamp(mu_sig, min=mu_bg + 1e-8)
        r_bg = torch.clamp(r_bg, min=1e-6)
        r_sig = torch.clamp(r_sig, min=1e-6)

        return {'mu_bg': mu_bg, 'mu_sig': mu_sig, 'r_bg': r_bg, 'r_sig': r_sig, 'pi_sig': pi_sig}

    def per_tag_loglik(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if counts is None:
            raise ValueError("NBModel.per_tag_loglik requires counts.")
        return _nb_logpmf_torch(counts, params['mu_sig'], params['r_sig'])

    def _per_tag_loglik_bg(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if counts is None:
            raise ValueError("NBModel._per_tag_loglik_bg requires counts.")
        return _nb_logpmf_torch(counts, params['mu_bg'], params['r_bg'])

    def fit_params(self, X: Optional[torch.Tensor], counts: Optional[torch.Tensor], post: torch.Tensor,
                   configs: torch.Tensor, params_old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if counts is None:
            raise ValueError("NBModel.fit_params requires counts.")
        device = counts.device
        configs = configs.to(device=device, dtype=torch.float32)
        q = post.to(device=device, dtype=torch.float32)

        S = q.matmul(configs)           # (N,H)
        B = q.matmul(1.0 - configs)

        sum_s = torch.clamp(S.sum(dim=0), min=EPS)
        sum_b = torch.clamp(B.sum(dim=0), min=EPS)

        sum_s_x = (S * counts).sum(dim=0)
        sum_s_x2 = (S * (counts * counts)).sum(dim=0)
        sum_b_x = (B * counts).sum(dim=0)
        sum_b_x2 = (B * (counts * counts)).sum(dim=0)

        mu_sig = sum_s_x / sum_s
        var_sig = (sum_s_x2 / sum_s) - (mu_sig * mu_sig)
        mu_bg = sum_b_x / sum_b
        var_bg = (sum_b_x2 / sum_b) - (mu_bg * mu_bg)

        r_sig = torch.clamp((mu_sig * mu_sig) / torch.clamp(var_sig - mu_sig, min=1e-8), min=1e-8)
        r_bg = torch.clamp((mu_bg * mu_bg) / torch.clamp(var_bg - mu_bg, min=1e-8), min=1e-8)

        mu_sig = torch.clamp(mu_sig, min=1e-8)
        mu_bg = torch.clamp(mu_bg, min=1e-8)
        pi_sig = sum_s / (sum_s + sum_b + EPS)

        return {'mu_bg': mu_bg, 'mu_sig': mu_sig, 'r_bg': r_bg, 'r_sig': r_sig, 'pi_sig': pi_sig}
