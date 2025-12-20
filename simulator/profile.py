import numpy as np

class ProfileRegistry:
    def __init__(self, h_tags, mu_high=7.0, mu_low=1.0, ambient_level=0.1, seed=42):
        self.rng = np.random.default_rng(seed)
        self.h_tags = h_tags
        self.htos = [f"HTO_sim_{i+1:02d}" for i in range(h_tags)]
        
        self.profiles = {
            hto: {
                "mu_sig": mu_high + self.rng.normal(0, 0.2),
                "mu_bg": mu_low + self.rng.normal(0, 0.1),
                "sigma_sig": 0.5,
                "sigma_bg": 0.6,
                "r_dispersion": self.rng.uniform(1.5, 3.0)  # 离散度参数
            } for hto in self.htos
        }

        # 全局背景向量
        self.ambient_vector = np.full(h_tags, ambient_level) + self.rng.uniform(0, 0.05, size=h_tags)