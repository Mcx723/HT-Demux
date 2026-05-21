import numpy as np


class ProfileRegistry:
    """
    Registry of per-HTO signal profiles.
    
    This class defines the biological signal characteristics
    (mean expression levels, noise scales, and ambient background)
    shared by different statistical engines (e.g., GMM, NB).
    """

    def __init__(
        self,
        h_tags: int,
        mu_high: float = 6.0,
        mu_low: float = 1.0,
        ambient_level: float = 0.1,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.h_tags = h_tags

        # HTO channel names
        self.htos = [f"HTO_sim_{i+1:02d}" for i in range(h_tags)]

        # Per-HTO biological signal profiles
        self.profiles = {
            hto: {
                "mu_sig": mu_high + self.rng.normal(0, 0.2),
                "mu_bg": mu_low + self.rng.normal(0, 0.1),
                "sigma_sig": 0.5,
                "sigma_bg": 0.6,
            }
            for hto in self.htos
        }

        # Global ambient background vector
        self.ambient_vector = (
            np.full(h_tags, ambient_level)
            + self.rng.uniform(0, 0.05, size=h_tags)
        )
