import torch
import numpy as np
from abc import ABC, abstractmethod


class CellEngine(ABC):
    """
    Base cell engine.

    Responsible for sampling the latent biological signal space
    (log-scale HTO intensities) shared by different observation models.
    """

    def __init__(self, registry, seed: int = 42):
        self.registry = registry
        self.rng = np.random.default_rng(seed)

        # ---- realism knobs (shared by GMM & NB) ----
        self.cell_eff_sd = 0.5        # cell-level capture efficiency (log-space)
        self.ambient_cell_sd = 0.05   # ambient fluctuation per cell

        # donor capture failure (key realism factor)
        self.p_pos_given_donor = 0.9  # P(signal | true donor)

    def _sample_latent_space(self, n_cells: int):
        """
        Sample latent log-intensity matrix and donor assignments.

        Returns
        -------
        latent_matrix : torch.Tensor, shape (n_cells, n_htos)
            Latent log-scale signal matrix.
        donor_indices : np.ndarray, shape (n_cells,)
            Index of true positive HTO for each cell.
        """
        h = self.registry.h_tags

        donor_indices = self.rng.choice(h, size=n_cells)
        amb = torch.from_numpy(self.registry.ambient_vector).float()

        latent_matrix = torch.zeros((n_cells, h))

        # ---- cell-level multiplicative noise (log-space additive) ----
        cell_eff = torch.normal(
            mean=0.0,
            std=self.cell_eff_sd,
            size=(n_cells, 1)
        )

        for i, hto in enumerate(self.registry.htos):
            p = self.registry.profiles[hto]

            # ---- ambient: cell-wise fluctuation ----
            ambient_noise = torch.normal(
                mean=amb[i],
                std=self.ambient_cell_sd,
                size=(n_cells,)
            )

            # ---- background signal ----
            bg_noise = torch.normal(
                mean=p["mu_bg"],
                std=p["sigma_bg"],
                size=(n_cells,)
            )

            latent_matrix[:, i] = bg_noise + ambient_noise

            # ---- positive signal for donor cells ----
            is_donor = donor_indices == i
            n_pos = int(is_donor.sum())

            if n_pos > 0:
                # donor capture failure (Bernoulli)
                is_true_pos = torch.rand(n_pos) < self.p_pos_given_donor

                sig_noise = torch.normal(
                    mean=p["mu_sig"],
                    std=p["sigma_sig"],
                    size=(n_pos,)
                )

                bg_fallback = torch.normal(
                    mean=p["mu_bg"],
                    std=p["sigma_bg"],
                    size=(n_pos,)
                )

                mixed_signal = torch.where(
                    is_true_pos,
                    sig_noise,
                    bg_fallback
                )

                latent_matrix[is_donor, i] = (
                    mixed_signal + ambient_noise[is_donor]
                )

        # ---- apply cell efficiency (shared distortion) ----
        latent_matrix = latent_matrix + cell_eff

        return latent_matrix, donor_indices

    @abstractmethod
    def generate(self, n_cells: int):
        pass


class GMMCellEngine(CellEngine):
    """
    Continuous observation model.

    Outputs continuous log-scale signals,
    but includes experimental compression to avoid unrealistically
    perfect Gaussian mixtures.
    """

    def generate(self, n_cells: int):
        latent_matrix, donor_indices = self._sample_latent_space(n_cells)

        # ---- experimental projection (continuous, non-NB) ----
        latent_matrix = torch.log1p(torch.exp(latent_matrix))

        return latent_matrix, donor_indices


class NBCellEngine(CellEngine):
    """
    Discrete observation model.

    Projects latent log-intensities to count space using a
    Negative Binomial–like Gamma-Poisson mixture.
    """

    def __init__(self, registry, dispersion: float = 1.0, seed: int = 42):
        super().__init__(registry, seed=seed)
        self.dispersion = dispersion

    def generate(self, n_cells: int):
        latent_matrix, donor_indices = self._sample_latent_space(n_cells)

        # Mean parameter in count space
        mu = torch.exp(latent_matrix)

        # Gamma-Poisson (NB) mixture
        if self.dispersion > 0:
            gamma_shape = 1.0 / self.dispersion
            gamma_scale = mu * self.dispersion

            rate = torch.distributions.Gamma(
                concentration=gamma_shape,
                rate=1.0 / gamma_scale
            ).sample()
        else:
            rate = mu

        counts = torch.poisson(rate)

        return counts.int(), donor_indices
