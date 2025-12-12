# train.py
import torch
from typing import Dict, Optional

EPS = 1e-12


def e_step(
    model,
    X: Optional[torch.Tensor],
    counts: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    pi: torch.Tensor,
    configs: torch.Tensor
):
    """
    E-step:
      lognum(n,k) = loglik(n,k) + log pi(k)
      post = softmax(lognum)
    Returns:
      post: (N,K) tensor
      total_ll: float (sum log-likelihood)
    """
    loglik_nk = model.loglik_model(X, counts, params, configs=configs)

    log_pi = torch.log(torch.clamp(pi, min=1e-12))      # (K,)
    lognum = loglik_nk + log_pi.view(1, -1)             # (N,K)

    maxv = lognum.max(dim=1, keepdim=True).values
    ex = torch.exp(lognum - maxv)
    denom = ex.sum(dim=1, keepdim=True) + EPS
    post = ex / denom                                   # (N,K)

    total_ll = (maxv + torch.log(denom)).sum().item()

    return post, total_ll


def m_step(
    model,
    X: Optional[torch.Tensor],
    counts: Optional[torch.Tensor],
    post: torch.Tensor,
    configs: torch.Tensor,
    params_old: Dict[str, torch.Tensor]
):
    """
    M-step: update π and model parameters via model.fit_params.
    """
    device = post.device
    N, K = post.shape

    # 1) update π
    pi = post.sum(dim=0) / float(N)
    pi = torch.clamp(pi, min=1e-12)
    pi = pi / pi.sum()              # (K,)

    # 2) model-specific updates
    params_new = model.fit_params(X, counts, post, configs, params_old)

    return pi, params_new


def run_em(
    model,
    X: Optional[torch.Tensor],
    counts: Optional[torch.Tensor],
    configs: torch.Tensor,
    init_params: Dict[str, torch.Tensor],
    init_pi: Optional[torch.Tensor] = None,
    init_post: Optional[torch.Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-5
):
    """
    Run EM and return dictionary with fitted params, pi, post, trace.
    """
    device = configs.device
    K = configs.shape[0]

    if init_pi is not None:
        pi = init_pi.to(device=device, dtype=torch.float32)
    elif init_post is not None:
        pi = init_post.sum(dim=0).to(device) / float(init_post.shape[0])
        pi = pi / pi.sum()
    else:
        pi = torch.ones(K, device=device) / float(K)

    params = {k: v.clone().to(device) for k, v in init_params.items()}

    loglik_trace = []
    prev_ll = -1e30

    # EM iterations
    for it in range(1, max_iter + 1):
        post, ll = e_step(model, X, counts, params, pi, configs)
        loglik_trace.append(ll)

        pi, params = m_step(model, X, counts, post, configs, params)

        if it > 1 and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return {
        "pi": pi,
        "params": params,
        "post": post,
        "loglik_trace": loglik_trace,
        "n_iter": it
    }
