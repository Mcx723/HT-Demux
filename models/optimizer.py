import torch
from typing import Optional
from torch.optim.adam import Adam  # Explicit import to satisfy Pylance
from .paramshared_cluster import ParamSharedCluster
from .base_cluster import GMMCluster, NBCluster

EPS = 1e-12

@torch.no_grad()
def fit_em(model: ParamSharedCluster, 
           X: torch.Tensor, 
           counts: Optional[torch.Tensor] = None, 
           max_iter: int = 50, 
           tol: float = 1e-4) -> torch.Tensor:
    """
    Expectation-Maximization algorithm for cluster parameter estimation.
    """
    model.base_cluster.init_parameters(X, counts)
    prev_ll = -float('inf')
    
    for i in range(max_iter):
        log_joint = model.calculate_posterior(X, counts)
        curr_ll = torch.logsumexp(log_joint, dim=1).mean().item()
        
        post = model.posteriors
        if post is None:
            break
        
        # M-step: Update Mixing Proportions
        model.pi_logits.data.copy_(torch.log(post.sum(0) / post.shape[0] + EPS))
        
        # M-step: Update Distribution Means
        S, B = post @ model.configs, post @ (1.0 - model.configs)
        sum_s, sum_b = S.sum(0) + EPS, B.sum(0) + EPS
        base = model.base_cluster
        
        target = counts if isinstance(base, NBCluster) and counts is not None else X
        base.mu_sig.data.copy_((S * target).sum(0) / sum_s)
        base.mu_bg.data.copy_((B * target).sum(0) / sum_b)
        
        if abs(curr_ll - prev_ll) < tol:
            break
        prev_ll = curr_ll
        
    return torch.argmax(model.posteriors, dim=1) if model.posteriors is not None else torch.zeros(X.shape[0], device=X.device)

def fit_gd(model: ParamSharedCluster, 
           X: torch.Tensor, 
           counts: Optional[torch.Tensor] = None, 
           lr: float = 0.01, 
           max_iter: int = 100) -> torch.Tensor:
    """
    Gradient Descent optimization using the Adam optimizer.
    """
    model.base_cluster.init_parameters(X, counts)
    
    # Using the explicitly imported Adam class
    optimizer = Adam(model.parameters(), lr=lr)
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        log_joint = model.calculate_posterior(X, counts)
        loss = -torch.mean(torch.logsumexp(log_joint, dim=1))
        loss.backward()
        optimizer.step()
        
    return torch.argmax(model.posteriors, dim=1) if model.posteriors is not None else torch.zeros(X.shape[0], device=X.device)