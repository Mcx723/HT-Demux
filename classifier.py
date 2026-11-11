import torch
import numpy as np
import pandas as pd
from typing import List, Optional


class HTClassifier:
    """Interpret the final shared-model posterior; never hard-label EM inputs."""

    def __init__(self, cfg_labels: List[str], tag_names: List[str], configs: torch.Tensor):
        self.cfg_labels = cfg_labels
        self.tag_names = tag_names
        self.configs = configs
        self.cfg_sums = configs.sum(dim=1).cpu().numpy()

    def classify(self, posteriors: torch.Tensor, barcodes: np.ndarray,
                 map_thresh: float = 0.9, decision_mode: str = 'hierarchical',
                 marginal_singlet_thresh: float = 0.05,
                 marginal_doublet_thresh: float = 0.99,
                 marginal_batch_size: int = 4096,
                 compute_device: Optional[str] = None,
                 posterior_thresh: float = 0.5,
                 multiplet_prior_scale: float = 0.5) -> pd.DataFrame:
        if posteriors is None:
            raise ValueError('Posteriors not found. Run model first.')
        if decision_mode not in {'hierarchical', 'posterior', 'map', 'marginal', 'joint'}:
            raise ValueError('Unknown decision mode')
        if marginal_batch_size < 1 or len(posteriors) != len(barcodes):
            raise ValueError('Invalid batch size or barcode length')
        for threshold in [map_thresh, marginal_singlet_thresh, marginal_doublet_thresh, posterior_thresh]:
            if not 0 <= threshold <= 1:
                raise ValueError('Probability thresholds must be in [0, 1]')
        if not np.isfinite(multiplet_prior_scale) or multiplet_prior_scale <= 0:
            raise ValueError('Multiplet prior scale must be finite and greater than zero')
        device = posteriors.device if compute_device is None else torch.device(compute_device)
        configs = self.configs.to(device)
        cfg_np = self.configs.cpu().numpy()
        single_ids = np.flatnonzero(self.cfg_sums == 1)
        multi_ids = np.flatnonzero(self.cfg_sums >= 2)
        negative_ids = np.flatnonzero(self.cfg_sums == 0)
        single_tags = cfg_np[single_ids].argmax(1)
        bio = np.array(['Unassigned' if k == 0 else
                        self.tag_names[int(cfg_np[i].argmax())] if k == 1 else 'Multiplet'
                        for i, k in enumerate(self.cfg_sums)], dtype=object)
        parts = []
        for start in range(0, len(posteriors), marginal_batch_size):
            stop = min(start + marginal_batch_size, len(posteriors))
            block = posteriors[start:stop]
            p = block.detach().cpu().numpy()
            if not np.isfinite(p).all() or (p < -1e-6).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
                raise ValueError('Invalid normalized posterior')
            if decision_mode in {'hierarchical', 'posterior', 'marginal'} and multiplet_prior_scale != 1.0:
                weights = np.ones(len(self.cfg_sums), dtype=p.dtype)
                weights[multi_ids] = multiplet_prior_scale
                p = p * weights[None, :]
                p /= p.sum(1, keepdims=True)
            n = len(p); row = np.arange(n)
            raw = p.argmax(1); map_probability = p[row, raw]
            marg = (block.to(device) @ configs).detach().cpu().numpy().clip(0, 1)
            single = p[:, single_ids]
            top_single = single.argmax(1)
            single_prob = single[row, top_single]
            p_single = single.sum(1)
            p_negative = p[:, negative_ids].sum(1) if len(negative_ids) else np.zeros(n)
            p_multi = np.clip(p.sum(1) - p_single - p_negative, 0, 1)
            if decision_mode == 'hierarchical':
                class_probs = np.column_stack([p_single, p_multi, p_negative])
                category = class_probs.argmax(1)
                category_probability = class_probs[row, category]
                identity_probability = np.divide(
                    single_prob, p_single, out=np.zeros(n), where=p_single > 0
                ).clip(0, 1)
                final = np.asarray(self.tag_names, dtype=object)[single_tags[top_single]].copy()
                final[category == 1] = 'Multiplet'
                final[category == 2] = 'Unassigned'
                call_type = np.asarray(['Singlet', 'Multiplet', 'Negative'], dtype=object)[category]
                identity = final.copy()
                identity[category != 0] = ''
                selected_probability = np.where(category == 0, single_prob, np.nan)
                identity_probability[category != 0] = np.nan
                fields = dict(
                    barcode=barcodes[start:stop], assignment_raw=final.copy(),
                    assignment=final.copy(), assignment_final=final,
                    assignment_type=call_type, probability=category_probability,
                    map_probability=category_probability, p_multiplet=p_multi,
                    p_singlet=p_single, p_negative=p_negative,
                    assignment_tags=identity, identity_probability=selected_probability,
                    identity_conditional_probability=identity_probability,
                )
                for i, tag in enumerate(self.tag_names):
                    fields[f'p_{tag}'] = marg[:, i]
                parts.append(pd.DataFrame(fields))
                continue
            # Biological category probabilities form a partition: each specific
            # singlet, any multiplet, and background. Marginals alone do not.
            class_probs = np.column_stack([single, p_multi, p_negative])
            winner = class_probs.argmax(1)
            confidence = class_probs[row, winner]
            is_multi = winner == len(single_ids)
            is_background = winner == len(single_ids) + 1
            final = np.asarray(self.tag_names, dtype=object)[single_tags[top_single]].copy()
            final[is_multi] = 'Multiplet'
            runner_up = np.partition(class_probs, -2, axis=1)[:, -2]
            tied = np.isclose(confidence, runner_up, atol=1e-7, rtol=0)
            final[is_background | (confidence < posterior_thresh) | tied] = 'Unassigned'
            chosen = single_ids[top_single].copy()
            need_multi = is_multi.copy()
            if decision_mode == 'map':
                final = bio[raw].copy()
                final[map_probability < map_thresh] = 'Unassigned'
                chosen = raw.copy(); confidence = map_probability
                need_multi = np.zeros(n, dtype=bool)
            elif decision_mode == 'joint':
                final = bio[raw].copy()
                top_two = np.partition(p, -2, axis=1)[:, -2]
                final[np.isclose(map_probability, top_two, atol=1e-7, rtol=0)] = 'Unassigned'
                chosen = raw.copy(); confidence = map_probability
                need_multi = np.zeros(n, dtype=bool)
            elif decision_mode == 'marginal':
                order = np.argsort(-marg, axis=1)
                top = marg[row, order[:, 0]]
                second = marg[row, order[:, 1]] if marg.shape[1] > 1 else np.zeros(n)
                final = np.asarray(self.tag_names, dtype=object)[order[:, 0]].copy()
                final[second >= marginal_doublet_thresh] = 'Multiplet'
                final[top < marginal_singlet_thresh] = 'Unassigned'
                chosen = single_ids[top_single].copy()
                need_multi = final == 'Multiplet'
                confidence = map_probability  # explicit legacy behavior
            # Reuse the already computed MAP whenever it is a multiplet.
            # Only ambiguous type calls need a second scan over pair identities.
            use_raw = need_multi & (self.cfg_sums[raw] >= 2)
            chosen[use_raw] = raw[use_raw]
            remaining = np.flatnonzero(need_multi & ~use_raw)
            if len(remaining) and len(multi_ids):
                chosen[remaining] = multi_ids[p[np.ix_(remaining, multi_ids)].argmax(1)]
            selected_prob = p[row, chosen]
            denominator = np.where(final == 'Multiplet', p_multi, p_single)
            identity_confidence = np.divide(selected_prob, denominator,
                                            out=np.zeros(n), where=denominator > 0).clip(0, 1)
            identity = np.asarray(self.cfg_labels, dtype=object)[chosen].copy()
            # A type call can be certain while the identities remain ambiguous.
            identity[(final == 'Unassigned') | ((identity_confidence < posterior_thresh) & (decision_mode != 'joint'))] = ''
            fields = dict(
                barcode=barcodes[start:stop], assignment_raw=np.asarray(self.cfg_labels)[raw],
                assignment=bio[raw], assignment_final=final, probability=confidence,
                map_probability=map_probability, p_multiplet=p_multi,
                p_singlet=p_single, p_negative=p_negative,
                assignment_tags=identity, identity_probability=selected_prob,
                identity_conditional_probability=identity_confidence)
            for i, tag in enumerate(self.tag_names):
                fields[f'p_{tag}'] = marg[:, i]
            parts.append(pd.DataFrame(fields))
        return pd.concat(parts, ignore_index=True)
