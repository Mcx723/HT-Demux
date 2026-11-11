import torch
from typing import Optional
from torch.optim.adam import Adam  # Explicit import to satisfy Pylance
from .paramshared_cluster import ParamSharedCluster
from .base_cluster import GMMCluster, NBCluster

EPS = 1e-12

@torch.no_grad()
def _final_posteriors(model, X, counts, batch_size, posterior_device):
    destination = X.device if posterior_device is None else torch.device(posterior_device)
    posterior = torch.empty((len(X), model.K), dtype=X.dtype, device=destination)
    total_ll = 0.0
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        joint = model.calculate_posterior(X[start:stop], None if counts is None else counts[start:stop])
        if not torch.isfinite(joint).all():
            raise FloatingPointError('Nonfinite final joint likelihood')
        total_ll += torch.logsumexp(joint, dim=1).sum(dtype=torch.float64).item()
        posterior[start:stop].copy_(model.posteriors.to(destination))
    model.posteriors = posterior
    return total_ll / len(X)

class _NBHistograms:
    """Exact weighted count histograms for the shared NB M-step (no labels/GT)."""
    def __init__(self, counts, batch_size):
        import numpy as np
        if not torch.isfinite(counts).all() or (counts < 0).any() or (counts != counts.round()).any():
            raise ValueError('NB EM requires finite nonnegative integer counts')
        self.counts = counts
        self.inverse = None
        h = counts.shape[1]
        bins = int(counts.max().item()) + 1
        if h * bins > 32_000_000:
            self.values, self.inverse = torch.unique(counts, sorted=True, return_inverse=True)
            bins = len(self.values)
        else:
            self.values = torch.arange(bins, device=counts.device, dtype=counts.dtype)
        self.total = torch.zeros((h, bins), dtype=torch.float64, device=counts.device)
        for start in range(0, len(counts), batch_size):
            idx = self.index(start, min(start + batch_size, len(counts)))
            self.total.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
        self.total_np = self.total.cpu().numpy()
        self.values_np = self.values.double().cpu().numpy()
        self.support = [np.flatnonzero(row) for row in self.total_np]

    def index(self, start, stop):
        source = self.counts if self.inverse is None else self.inverse
        return source[start:stop].T.long()

    @torch.no_grad()
    def update(self, base, signal):
        import numpy as np
        from scipy.optimize import brentq
        from scipy.special import digamma, gammaln
        sig = signal.cpu().numpy()
        gains = 0.0
        for kind, hist in [('sig', sig), ('bg', np.maximum(self.total_np - sig, 0.0))]:
            mu_param = getattr(base, 'mu_' + kind)
            r_param = getattr(base, 'log_r_' + kind)
            old_mu = mu_param.detach().double().cpu().numpy()
            old_r = np.exp(r_param.detach().double().cpu().numpy())
            means, shapes = old_mu.copy(), old_r.copy()
            for h, support in enumerate(self.support):
                x, w = self.values_np[support], hist[h, support]
                mass = w.sum()
                if mass < 1e-6:
                    continue
                w = w / mass
                mu = max(float(w @ x), 1e-8)
                factorial = gammaln(x + 1.0)
                def objective(r, mean=mu):
                    ll = (gammaln(x + r) - gammaln(r) - factorial
                          - r * np.log1p(mean / r)
                          + x * (np.log(mean) - np.log(r + mean)))
                    return float(w @ ll)
                def score(log_r):
                    r = np.exp(log_r)
                    return float(w @ (digamma(x + r) - digamma(r)) - np.log1p(mu / r))
                lower = base.signal_shape_min if kind == 'sig' else 1e-3
                lo, hi = np.log(lower), np.log(1e6)
                candidates = [old_r[h], np.exp(lo), np.exp(hi)]
                if score(lo) * score(hi) < 0:
                    candidates.append(np.exp(brentq(score, lo, hi, xtol=1e-7)))
                # Including the old shape guarantees a nondecreasing weighted Q.
                best = max(candidates, key=objective)
                means[h], shapes[h] = mu, best
                gains += mass * (objective(best) - objective(old_r[h], max(old_mu[h], 1e-8)))
            mu_param.copy_(torch.as_tensor(means, device=mu_param.device, dtype=mu_param.dtype))
            r_param.copy_(torch.as_tensor(np.log(shapes), device=r_param.device, dtype=r_param.dtype))
        return gains / len(self.counts)


@torch.no_grad()
def fit_em(model: ParamSharedCluster,
           X: torch.Tensor,
           counts: Optional[torch.Tensor] = None,
           max_iter: int = 100,
           tol: float = 1e-4,
           batch_size: int = 4096,
           posterior_device: Optional[str] = None,
           min_iter: int = 10) -> torch.Tensor:
    """Soft EM: aggregate all configurations before updating shared tag parameters.

    NB shapes maximize the responsibility-weighted likelihood numerically; they
    are not fixed or estimated from hard singlet/doublet calls. Histograms retain
    every observed integer count and its soft weight, without subsampling.
    """
    if len(X) == 0 or max_iter < 1 or batch_size < 1 or tol < 0 or min_iter < 1:
        raise ValueError('EM requires nonempty data and positive iteration/batch limits')
    base = model.base_cluster
    base.init_parameters(X, counts)
    if isinstance(base, NBCluster) and counts is None:
        raise ValueError('NB EM requires counts')
    nb_stats = _NBHistograms(counts, batch_size) if isinstance(base, NBCluster) else None
    history, parameter_trace, mstep_gains = [], [], []
    converged, stable_steps = False, 0
    previous_pi = model.log_pi.exp().clone()
    for i in range(max_iter):
        mixing = torch.zeros(model.K, device=X.device, dtype=torch.float64)
        if nb_stats is not None:
            signal_hist = torch.zeros_like(nb_stats.total)
        else:
            sum_s = torch.zeros(model.H, device=X.device, dtype=torch.float64)
            weighted_s = torch.zeros_like(sum_s)
            square_s = torch.zeros_like(sum_s)
        ll_sum = 0.0
        old_params = {name: p.clone() for name, p in base.named_parameters()}
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            xb = X[start:stop]
            cb = None if counts is None else counts[start:stop]
            joint = model.calculate_posterior(xb, cb)
            if not torch.isfinite(joint).all():
                raise FloatingPointError('Nonfinite EM joint likelihood')
            ll_sum += torch.logsumexp(joint, dim=1).sum(dtype=torch.float64).item()
            post = model.posteriors
            mixing += post.sum(0, dtype=torch.float64)
            signal_weight = (post @ model.configs).clamp(0.0, 1.0)
            if nb_stats is not None:
                signal_hist.scatter_add_(1, nb_stats.index(start, stop), signal_weight.T.double())
            else:
                sw, target = signal_weight.double(), xb.double()
                sum_s += sw.sum(0)
                weighted_s += (sw * target).sum(0)
                square_s += (sw * target.square()).sum(0)
        curr_ll = ll_sum / len(X)
        history.append(curr_ll)
        model.pi_logits.copy_(torch.log(mixing / mixing.sum() + EPS))
        if nb_stats is not None:
            mstep_gains.append(nb_stats.update(base, signal_hist))
        else:
            target = X.double()
            for kind, mass, first, second in [
                ('sig', sum_s, weighted_s, square_s),
                ('bg', len(X) - sum_s, target.sum(0) - weighted_s,
                 target.square().sum(0) - square_s),
            ]:
                active = mass > 1e-6
                mean = first / mass.clamp_min(EPS)
                variance = (second / mass.clamp_min(EPS) - mean.square()).clamp_min(1e-4)
                getattr(base, 'mu_' + kind)[active] = mean[active].float()
                getattr(base, 'log_sigma_' + kind)[active] = .5 * variance[active].log().float()
        pi = model.log_pi.exp()
        pi_change = (pi - previous_pi).abs().sum().item()
        previous_pi = pi.clone()
        changes = []
        for name, p in base.named_parameters():
            old = old_params[name]
            delta = (p - old).abs() if name.startswith('log_') else (p - old).abs() / old.abs().clamp_min(1.0)
            changes.append(delta.max().item())
        parameter_change = max(changes)
        state = {'iteration': i + 1, 'mean_log_likelihood': curr_ll,
                 'mixing_l1_change': pi_change, 'max_parameter_change': parameter_change,
                 'mu_sig': base.mu_sig.cpu().tolist(), 'mu_bg': base.mu_bg.cpu().tolist()}
        if nb_stats is not None:
            state.update(r_sig=base.log_r_sig.exp().cpu().tolist(), r_bg=base.log_r_bg.exp().cpu().tolist())
        parameter_trace.append(state)
        if len(history) > 1 and curr_ll < history[-2] - max(5e-4, abs(curr_ll) * 2e-7):
            raise FloatingPointError('EM likelihood decreased beyond numerical tolerance')
        stable = (len(history) > 1 and abs(curr_ll - history[-2]) < tol
                  and pi_change < 1e-3 and parameter_change < 1e-3)
        stable_steps = stable_steps + 1 if stable else 0
        if i + 1 >= min(min_iter, max_iter) and stable_steps >= 3:
            converged = True
            break
    final_ll = _final_posteriors(model, X, counts, batch_size, posterior_device)
    if final_ll < history[-1] - max(5e-4, abs(final_ll) * 2e-7):
        raise FloatingPointError('Final EM likelihood decreased beyond numerical tolerance')
    model.fit_diagnostics = dict(
        optimizer='EM', iterations=len(history), max_iter=max_iter, min_iter=min_iter,
        converged=converged, stop_reason='likelihood_and_parameters_stable' if converged else 'max_iter',
        tolerance=tol, batch_size=batch_size, final_mean_log_likelihood=final_ll,
        mean_log_likelihood_trace=history+[final_ll], parameter_trace=parameter_trace,
        nb_weighted_mstep_gain=mstep_gains, posterior_device=str(model.posteriors.device),
        nb_shape_bounds={'signal': [base.signal_shape_min, 1e6], 'background': [1e-3, 1e6]} if nb_stats is not None else None,
        initialization='tag_count_adaptive_tail_and_median_background; equal_mass_per_order',
        distribution_update='shared_means_and_weighted_MLE_NB_shapes' if nb_stats is not None else 'shared_means_and_variances')
    return torch.argmax(model.posteriors, dim=1)


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
        
    with torch.no_grad():
        model.calculate_posterior(X, counts)
    return torch.argmax(model.posteriors, dim=1)


# Implicit all-order shared inference, using the existing optimizer module.
"""Exact implicit inference over all tag sets with a learned, structured prior.

The nonempty tag set follows independent Bernoulli draws conditioned on at least
one positive tag; a separate learned mass represents all-background droplets.
Signal/background NB parameters remain shared across every configuration.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import brentq
from .base_cluster import NBCluster
from .base_cluster import DiscreteLogNormalCluster


def _log_nonempty(log_none, logits=None):
    out = torch.log(-torch.expm1(log_none))
    if logits is not None:
        # When all odds underflow, multiple simultaneous positives are negligible.
        out = torch.where(torch.isfinite(out), out, torch.logsumexp(logits, dim=1))
    return out


def conditional_bernoulli_mstep(marginal):
    """MLE for Bernoulli parameters after conditioning on a nonempty set."""
    m = np.clip(np.asarray(marginal, dtype=np.float64), 1e-12, 1 - 1e-12)
    if m.sum() <= 1 + 1e-8:
        # Boundary limit: the nonempty distribution becomes categorical.
        return np.maximum(m / m.sum() * 1e-6, 1e-12)
    def equation(z):
        # Divide by z so the excluded zero root cannot be selected numerically.
        if z == 0:
            return m.sum() - 1
        return -np.expm1(np.log1p(-m * z).sum()) / z - 1
    z = brentq(equation, 0., 1., xtol=1e-13)
    return np.clip(m * z, 1e-12, 1 - 1e-12)


class FactorizedSharedCluster(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_cluster = base
        self.H = base.num_features
        self.register_buffer('theta', torch.full((self.H,), min(.5, 1 / self.H), dtype=torch.float64))
        self.register_buffer('negative_mass', torch.tensor(.02, dtype=torch.float64))
        self.posteriors = None  # No N x 2^H tensor is constructed.

    def posterior(self, X, counts=None):
        ll = self.base_cluster.compute_probability(X, counts)
        diff = ll['sig'].double() - ll['bg'].double()
        log_theta = self.theta.log()
        log_one_minus_theta = torch.log1p(-self.theta)
        logits = diff + log_theta - log_one_minus_theta
        log_none_q = -F.softplus(logits).sum(1)
        log_nonempty_q = _log_nonempty(log_none_q, logits)
        log_nonempty_prior = _log_nonempty(log_one_minus_theta.sum())
        bg = ll['bg'].double().sum(1)
        positive_joint = (bg + log_one_minus_theta.sum() - log_none_q
                          + log_nonempty_q - log_nonempty_prior
                          + torch.log1p(-self.negative_mass))
        negative_joint = bg + self.negative_mass.log()
        evidence = torch.logaddexp(positive_joint, negative_joint)
        log_positive = positive_joint - evidence
        log_negative = negative_joint - evidence
        marginal = torch.exp(log_positive[:, None] + F.logsigmoid(logits) - log_nonempty_q[:, None]).clamp(0, 1)
        single = torch.exp(log_positive[:, None] + log_none_q[:, None] + logits - log_nonempty_q[:, None]).clamp(0, 1)
        negative = log_negative.exp().clamp(0, 1)
        positive = log_positive.exp().clamp(0, 1)
        multi = (positive - single.sum(1)).clamp(0, 1)
        return dict(log_evidence=evidence, marginal=marginal, single=single,
                    negative=negative, positive=positive, multi=multi,
                    logits=logits, log_none_q=log_none_q,
                    log_nonempty_q=log_nonempty_q, log_positive=log_positive)

    @torch.no_grad()
    def classify(self, counts, barcodes, tag_names, batch_size=4096,
                 posterior_thresh=.5, map_thresh=.9, decision_mode='hierarchical',
                 marginal_singlet_thresh=.05, marginal_doublet_thresh=.99,
                 include_tag_probabilities=True, multiplet_prior_scale=.5):
        if decision_mode not in ['hierarchical', 'posterior', 'map', 'marginal', 'joint']:
            raise ValueError('Unknown decision mode')
        if len(counts) != len(barcodes) or len(tag_names) != self.H:
            raise ValueError('Counts, barcodes and tags must align')
        if not all(0 <= t <= 1 for t in [posterior_thresh, map_thresh, marginal_singlet_thresh, marginal_doublet_thresh]):
            raise ValueError('Probability thresholds must be in [0,1]')
        if not np.isfinite(multiplet_prior_scale) or multiplet_prior_scale <= 0:
            raise ValueError('Multiplet prior scale must be finite and greater than zero')
        parts = []
        tags = np.asarray(tag_names, dtype=object)
        for start in range(0, len(counts), batch_size):
            c = counts[start:start + batch_size].to(self.theta.device)
            p = self.posterior(c, c)
            single_base = p['single'].cpu().numpy()
            marginal_base = p['marginal'].cpu().numpy()
            pm_base = p['multi'].cpu().numpy()
            pn_base = p['negative'].cpu().numpy()
            if decision_mode in {'hierarchical', 'posterior', 'marginal'} and multiplet_prior_scale != 1.0:
                norm = (single_base.sum(1) + multiplet_prior_scale * pm_base + pn_base)
                single = single_base / norm[:, None]
                pm = multiplet_prior_scale * pm_base / norm
                pn = pn_base / norm
                marg = (single_base + multiplet_prior_scale *
                        (marginal_base - single_base)) / norm[:, None]
            else:
                single, marg, pm, pn = single_base, marginal_base, pm_base, pn_base
            n = len(c); row = np.arange(n)
            ps = single.sum(1)
            if decision_mode == 'hierarchical':
                cls = np.column_stack([ps, pm, pn])
                category = cls.argmax(1)
                confidence = cls[row, category]
                best_single = single.argmax(1)
                best_single_prob = single[row, best_single]
                identity_conf = np.divide(
                    best_single_prob, ps, out=np.zeros(n), where=ps > 0
                ).clip(0, 1)
                final = tags[best_single].copy()
                final[category == 1] = 'Multiplet'
                final[category == 2] = 'Unassigned'
                call_type = np.asarray(['Singlet', 'Multiplet', 'Negative'], dtype=object)[category]
                identity = final.copy()
                identity[category != 0] = ''
                selected_prob = np.where(category == 0, best_single_prob, np.nan)
                identity_conf[category != 0] = np.nan
                fields = dict(
                    barcode=barcodes[start:start+n], assignment_raw=final.copy(),
                    assignment=final.copy(), assignment_final=final,
                    assignment_type=call_type, probability=confidence,
                    map_probability=confidence, p_multiplet=pm, p_singlet=ps,
                    p_negative=pn, assignment_tags=identity,
                    identity_probability=selected_prob,
                    identity_conditional_probability=identity_conf,
                )
                if include_tag_probabilities:
                    for j, tag in enumerate(tags):
                        fields[f'p_{tag}'] = marg[:, j]
                parts.append(pd.DataFrame(fields))
                continue
            odds = p['logits'].cpu().numpy()
            best_single = single.argmax(1)
            cls = np.column_stack([single, pm, pn])
            win = cls.argmax(1); confidence = cls[row, win]
            final = tags[best_single].copy()
            final[win == self.H] = 'Multiplet'
            second = np.partition(cls, -2, axis=1)[:, -2]
            final[(win == self.H + 1) | (confidence < posterior_thresh)
                  | np.isclose(confidence, second, atol=1e-7, rtol=0)] = 'Unassigned'
            # Full-set MAP is analytical: include all positive log odds.
            best_set = odds > 0
            empty = ~best_set.any(1)
            best_set[row[empty], odds[empty].argmax(1)] = True
            base_logprob = (p['log_positive'] + p['log_none_q'] - p['log_nonempty_q']).cpu().numpy()
            best_prob = np.exp(base_logprob + (odds * best_set).sum(1)).clip(0, 1)
            raw_negative = pn > best_prob
            raw_sets = best_set.copy()
            raw_type = np.where(best_set.sum(1) > 1, 'Multiplet', tags[best_set.argmax(1)]).astype(object)
            raw_type[raw_negative] = 'Unassigned'
            map_probability = np.maximum(best_prob, pn)
            if decision_mode == 'map':
                final = raw_type.copy(); confidence = map_probability.copy()
                final[confidence < map_thresh] = 'Unassigned'
            elif decision_mode == 'joint':
                # Under full tag-set 0/1 loss, choose the most probable complete
                # set, not the sum of many mutually exclusive multiplet sets.
                # Its probability need not exceed .5 in a multiclass problem.
                final = raw_type.copy(); confidence = map_probability.copy()
                ambiguous = np.isclose(odds, 0., atol=1e-7, rtol=0).any(1)
                ambiguous |= np.isclose(best_prob, pn, atol=1e-7, rtol=0)
                final[ambiguous] = 'Unassigned'
            elif decision_mode == 'marginal':
                order = np.argsort(-marg, axis=1)
                final = tags[order[:, 0]].copy()
                if self.H > 1:final[marg[row, order[:, 1]] >= marginal_doublet_thresh] = 'Multiplet'
                final[marg[row, order[:, 0]] < marginal_singlet_thresh] = 'Unassigned'
                confidence = map_probability.copy()
            chosen = np.zeros_like(best_set)
            chosen[row, best_single] = True
            multi_rows = np.flatnonzero(final == 'Multiplet')
            if len(multi_rows):
                chosen[multi_rows] = best_set[multi_rows]
                need_two = multi_rows[chosen[multi_rows].sum(1) < 2]
                if self.H > 1 and len(need_two):
                    top_two = np.argsort(-odds[need_two], axis=1)[:, :2]
                    chosen[need_two] = False
                    chosen[need_two[:, None], top_two] = True
            if decision_mode in ['map', 'joint']:chosen = raw_sets.copy()
            selected = np.exp(base_logprob + (odds * chosen).sum(1)).clip(0, 1)
            denominator = np.where(final == 'Multiplet', pm, single.sum(1))
            identity_conf = np.divide(selected, denominator, out=np.zeros(n), where=denominator > 0).clip(0, 1)
            identities = ['+'.join(sorted(tags[mask])) for mask in chosen]
            identities = np.asarray(identities, dtype=object)
            identities[(final == 'Unassigned') | ((identity_conf < posterior_thresh) & (decision_mode != 'joint'))] = ''
            raw_labels = np.asarray(['+'.join(tags[mask]) for mask in raw_sets], dtype=object)
            raw_labels[raw_negative] = 'Negative'
            fields = dict(barcode=barcodes[start:start+n], assignment_raw=raw_labels,
                          assignment=raw_type, assignment_final=final,
                          probability=confidence, map_probability=map_probability,
                          p_multiplet=pm, p_singlet=single.sum(1), p_negative=pn,
                          assignment_tags=identities, identity_probability=selected,
                          identity_conditional_probability=identity_conf)
            if include_tag_probabilities:
                for j, tag in enumerate(tags):fields[f'p_{tag}'] = marg[:, j]
            parts.append(pd.DataFrame(fields))
        return pd.concat(parts, ignore_index=True)


@torch.no_grad()
def fit_factorized_em(model, X, counts=None, max_iter=100, tol=1e-4,
                      batch_size=4096, min_iter=10, posterior_device=None,
                      init_tail_fraction=None):
    if counts is None or not isinstance(model.base_cluster, (NBCluster,DiscreteLogNormalCluster)):
        raise ValueError('Implicit structured EM requires a supported count emission')
    if len(X) == 0 or max_iter < 1 or batch_size < 1:
        raise ValueError('Nonempty counts and positive iteration/batch limits required')
    base = model.base_cluster
    if init_tail_fraction is not None:
        if not 0 < init_tail_fraction <= 0.25:
            raise ValueError('init_tail_fraction must be in (0, 0.25]')
        if not isinstance(base, NBCluster):
            raise ValueError('init_tail_fraction only applies to NB emissions')
        base.init_tail_fraction = float(init_tail_fraction)
    base.init_parameters(X, counts)
    stats = _NBHistograms(counts, batch_size)
    history=[];trace=[];gains=[];stable_steps=0;converged=False
    for iteration in range(max_iter):
        hist = torch.zeros_like(stats.total)
        positive_mass = 0.;ll_sum=0.
        old = {name: p.clone() for name,p in base.named_parameters()}
        old_theta=model.theta.clone();old_negative=model.negative_mass.clone()
        for start in range(0,len(X),batch_size):
            stop=min(start+batch_size,len(X));p=model.posterior(X[start:stop],counts[start:stop])
            if not torch.isfinite(p['log_evidence']).all():raise FloatingPointError('Nonfinite structured posterior')
            hist.scatter_add_(1,stats.index(start,stop),p['marginal'].T.double())
            positive_mass += p['positive'].sum().item()
            ll_sum += p['log_evidence'].sum().item()
        ll=ll_sum/len(X);history.append(ll)
        if positive_mass>1e-6:
            marg=hist.sum(1).cpu().numpy()/positive_mass
            theta=conditional_bernoulli_mstep(marg)
            model.theta.copy_(torch.as_tensor(theta,device=X.device))
        model.negative_mass.copy_(torch.as_tensor(np.clip(1-positive_mass/len(X),1e-10,1-1e-10),device=X.device))
        gains.append(base.weighted_mstep(stats,hist) if isinstance(base,DiscreteLogNormalCluster) else stats.update(base,hist))
        changes=[(model.theta-old_theta).abs().max().item(),abs((model.negative_mass-old_negative).item())]
        for name,p in base.named_parameters():
            changes.append(((p-old[name]).abs() if name.startswith('log_') else (p-old[name]).abs()/old[name].abs().clamp_min(1.)).max().item())
        change=max(changes)
        state=dict(iteration=iteration+1,mean_log_likelihood=ll,max_parameter_change=change,
                          negative_mass=model.negative_mass.item(),theta=model.theta.cpu().tolist(),
                          mu_sig=base.mu_sig.cpu().tolist(),mu_bg=base.mu_bg.cpu().tolist(),
                          **({'r_sig':base.log_r_sig.exp().cpu().tolist(),'r_bg':base.log_r_bg.exp().cpu().tolist()} if isinstance(base,NBCluster) else {'sigma_sig':base.log_sigma_sig.exp().cpu().tolist(),'sigma_bg':base.log_sigma_bg.exp().cpu().tolist()}))
        trace.append(state)
        if len(history)>1 and ll<history[-2]-max(5e-4,abs(ll)*2e-7):
            raise FloatingPointError('Structured EM likelihood decreased beyond numerical tolerance')
        stable=len(history)>1 and abs(ll-history[-2])<tol and change<1e-3
        stable_steps=stable_steps+1 if stable else 0
        if iteration+1>=min(min_iter,max_iter) and stable_steps>=3:converged=True;break
    final_ll=0.
    for start in range(0,len(X),batch_size):
        p=model.posterior(X[start:start+batch_size],counts[start:start+batch_size]);final_ll+=p['log_evidence'].sum().item()
    final_ll/=len(X)
    if final_ll<history[-1]-max(5e-4,abs(final_ll)*2e-7):raise FloatingPointError('Final structured EM likelihood decreased')
    model.fit_diagnostics=dict(optimizer='EM',configuration_prior='conditional_independent_Bernoulli_plus_background',
        inference='exact_implicit_all_tag_sets',iterations=len(history),max_iter=max_iter,min_iter=min_iter,
        converged=converged,stop_reason='likelihood_and_parameters_stable' if converged else 'max_iter',
        mean_log_likelihood_trace=history+[final_ll],final_mean_log_likelihood=final_ll,
        parameter_trace=trace,nb_weighted_mstep_gain=gains,negative_mass=model.negative_mass.item(),
        theta=model.theta.cpu().tolist(),emission_family='DLN' if isinstance(base,DiscreteLogNormalCluster) else 'NB',
        nb_shape_bounds={'signal':[base.signal_shape_min,1e6],'background':[1e-3,1e6]} if isinstance(base,NBCluster) else None,
        initialization={'signal_upper_tail_fraction':base.init_tail_fraction if isinstance(base,NBCluster) else None,
                         'default_rule':'adaptive max(1/H, 2/N), capped at 0.25' if isinstance(base,NBCluster) else None},
        batch_size=batch_size,posterior_storage='compact_tag_marginals_on_demand',
        distribution_update='shared_discrete_lognormal_mu_sigma_weighted_MLE' if isinstance(base,DiscreteLogNormalCluster) else 'shared_means_and_weighted_MLE_NB_shapes')
    return model
