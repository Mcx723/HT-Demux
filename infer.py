# infer.py
import torch
import pandas as pd
from typing import Tuple


@torch.no_grad()
def bayesian_infer(X: torch.Tensor, counts: torch.Tensor, model, params: dict, configs: torch.Tensor, pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute posterior and MAP prediction.

    Returns:
      post: (N,K) posterior probabilities
      pred: (N,) predicted cluster index
    """
    # log p(x | k)
    lg = model.loglik_model(X, counts, params, configs)   # (N,K)

    log_pi = torch.log(torch.clamp(pi, min=1e-12))                       # (K,)
    lg += log_pi.unsqueeze(0)

    # normalize
    maxv = lg.max(dim=1, keepdim=True).values
    post = torch.exp(lg - maxv)
    post = post / (post.sum(dim=1, keepdim=True) + 1e-12)

    pred = post.argmax(dim=1)

    return post, pred

