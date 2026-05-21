import torch
import numpy as np
import pandas as pd


class DropletFactory:
    """
    DropletFactory maps individual cells to droplets and aggregates HTO signals.

    This class represents the physical encapsulation process in droplet-based
    single-cell experiments and is independent of the statistical observation
    model (e.g., GMM or NB).
    """

    def __init__(self, m_droplets: int, seed: int = 42):
        """
        Parameters
        ----------
        m_droplets : int
            Total number of droplets to simulate.
        seed : int
            Random seed for reproducible cell-to-droplet assignment.
        """
        self.m_droplets = m_droplets
        self.rng = np.random.default_rng(seed)

    def produce(self, cell_matrix: torch.Tensor, donor_indices: np.ndarray, hto_names):
        """
        Aggregate cell-level HTO signals into droplets.

        Parameters
        ----------
        cell_matrix : torch.Tensor
            Cell-by-HTO matrix (continuous for GMM, integer counts for NB).
        donor_indices : np.ndarray
            Ground-truth donor index for each cell.
        hto_names : list[str]
            Names of HTO channels.

        Returns
        -------
        filtered_matrix : torch.Tensor
            Droplet-by-HTO aggregated matrix for non-empty droplets.
        final_gt : pandas.DataFrame
            Ground-truth droplet annotations.
        """
        n_cells, h_tags = cell_matrix.shape

        # 1. Physical encapsulation: random projection of cells to droplets
        droplet_map = self.rng.integers(
            low=0,
            high=self.m_droplets,
            size=n_cells
        )

        # 2. Vectorized aggregation (supports float or int tensors)
        droplet_matrix = torch.zeros(
            (self.m_droplets, h_tags),
            dtype=cell_matrix.dtype,
            device=cell_matrix.device
        )

        droplet_matrix.index_add_(
            0,
            torch.from_numpy(droplet_map).to(cell_matrix.device),
            cell_matrix
        )

        # 3. Construct ground-truth annotations
        df_cells = pd.DataFrame({
            "droplet_idx": droplet_map,
            "donor": [hto_names[i] for i in donor_indices],
        })

        gt = (
            df_cells
            .groupby("droplet_idx")["donor"]
            .apply(list)
            .to_frame()
        )

        gt["n_cells"] = gt["donor"].apply(len)
        gt["unique_donors"] = gt["donor"].apply(lambda x: len(set(x)))

        # 4. Remove empty droplets and reindex
        active_indices = gt.index.values
        filtered_matrix = droplet_matrix[active_indices]
        final_gt = gt.reset_index(drop=True)

        return filtered_matrix, final_gt

    @staticmethod
    def display_report(gt: pd.DataFrame):
        """
        Display a summary QC report of droplet composition.
        """
        total = len(gt)

        singlets = gt[gt["n_cells"] == 1]
        msm = gt[(gt["n_cells"] > 1) & (gt["unique_donors"] > 1)]
        ssm = gt[(gt["n_cells"] > 1) & (gt["unique_donors"] == 1)]

        print(f"\n{'='*15} HTO Simulation QC Report {'='*15}")
        print(f"Non-Empty Droplets: {total}")
        print("-" * 50)
        print(f"Singlets:           {len(singlets):<8} | {len(singlets)/total:>7.2%}")
        print(f"MSM (Heterotypic):  {len(msm):<8} | {len(msm)/total:>7.2%}")
        print(f"SSM (Homotypic):    {len(ssm):<8} | {len(ssm)/total:>7.2%}")
        print("-" * 50)

        if len(singlets) > 0:
            print("Singlet Distribution (Top 5 HTOs):")
            print(
                singlets["donor"]
                .apply(lambda x: x[0])
                .value_counts()
                .head(5)
            )

        print(f"{'='*50}\n")
