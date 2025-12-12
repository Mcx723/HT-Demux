# cluster.py
from typing import Optional, Tuple, Any
import torch


def init_cluster_centers(
    X: torch.Tensor,
    configs: torch.Tensor,
    mu_sig: Any,
    n_subset: int = 100,
    method: str = "closest"
) -> torch.Tensor:
    """
    Initialize cluster centers for each config.

    Args:
      X: (N,H) data matrix (must be a torch.Tensor)
      configs: (K,H) {0,1} float tensor
      mu_sig: (H,) tensor or ndarray (prototype signal mean)
      n_subset: number of nearest cells to average for center
      method: "closest" or "sum_mu"

    Returns:
      mu_init: (K,H) tensor
    """
    # normalize mu_sig -> torch tensor on same device as X
    if not torch.is_tensor(mu_sig):
        mu_sig = torch.tensor(mu_sig, dtype=torch.float32)
    mu_sig = mu_sig.to(dtype=torch.float32, device=X.device)

    N, H = X.shape
    K = configs.shape[0]
    configs = configs.to(dtype=torch.float32, device=X.device)

    mu_init = torch.zeros((K, H), dtype=torch.float32, device=X.device)

    # option: set mu for masked dims to mu_sig
    if method == "sum_mu":
        for k in range(K):
            mask = configs[k].bool()
            if mask.any():
                mu_init[k, mask] = mu_sig[mask]
        return mu_init

    # default: closest - find cells closest to prototype and average
    for k in range(K):
        mask = configs[k].bool()
        if not mask.any():
            continue
        X_sub = X[:, mask]                   # (N, h_k)
        target = mu_sig[mask]                # (h_k,)
        diff = X_sub - target.unsqueeze(0)   # (N, h_k)
        d2 = torch.sum(diff * diff, dim=1)   # (N,)

        npick = min(n_subset, N)
        if npick <= 0:
            continue
        vals, idx = torch.topk(-d2, npick)
        selected = X_sub[idx]                # (npick, h_k)
        mu_init[k, mask] = selected.mean(dim=0)

    return mu_init


def init_clusters(
    X: Optional[torch.Tensor],
    configs: torch.Tensor,
    mu_sig: Any,
    n_subset: int = 100,
    method: str = "closest",
    return_gamma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute (mu_init, pi_init, gamma_init).

    Args:
      X: (N,H) data tensor used for initialization.
      configs: (K,H) 0/1 cluster genotype matrix
      mu_sig: prototype per-tag signal (tensor or array-like)
      return_gamma: if True, return initial soft assignments gamma (N,K)

    Returns:
      - mu_init: (K,H) float tensor
      - pi: (K,) float tensor (sums to 1)
      - gamma: (N,K) float tensor if return_gamma True else None
    """
    # Accept Optional[X] for static typing; enforce at runtime
    if X is None:
        raise ValueError("init_clusters requires a data tensor X (got None). "
                         "Pass X (e.g., X_gmm or counts) to initialize clusters.")

    device = X.device
    configs = configs.to(device=device, dtype=torch.float32)

    mu_init = init_cluster_centers(X, configs, mu_sig, n_subset=n_subset, method=method)

    N, H = X.shape
    K = configs.shape[0]

    if return_gamma:
        # compute squared distances to each center, respecting masks
        d2 = torch.empty((N, K), dtype=torch.float32, device=device)
        for k in range(K):
            mask = configs[k].bool()
            if not mask.any():
                # if cluster masks nothing, set a large distance so softmax gives tiny weight
                d2[:, k] = float(1e12)
                continue
            X_sub = X[:, mask]
            center_sub = mu_init[k, mask].unsqueeze(0)
            diff = X_sub - center_sub
            d2[:, k] = torch.sum(diff * diff, dim=1)
        gamma = torch.softmax(-d2, dim=1)  # (N,K)
        pi = gamma.sum(dim=0) / float(N)
        pi = torch.clamp(pi, min=1e-12)
        pi = pi / pi.sum()
    else:
        gamma = None
        Kf = float(K)
        pi = torch.full((K,), 1.0 / Kf, dtype=torch.float32, device=device)

    return mu_init, pi, gamma
