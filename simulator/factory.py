import torch
import numpy as np
import pandas as pd

class DropletFactory:
    def __init__(self, m_droplets):
        self.m_droplets = m_droplets

    def produce(self, cell_matrix, donor_indices, hto_names):
        n_cells = cell_matrix.shape[0]
        h_tags = cell_matrix.shape[1]

        # 1. 物理分发 (随机投射)
        droplet_map = np.random.randint(0, self.m_droplets, size=n_cells)
        
        # 2. 向量化聚合：自动对齐 float (GMM) 或 int (NB)
        droplet_matrix = torch.zeros((self.m_droplets, h_tags), dtype=cell_matrix.dtype)
        droplet_matrix.index_add_(0, torch.from_numpy(droplet_map), cell_matrix)
        
        # 3. 真值构建
        df_cells = pd.DataFrame({
            'droplet_idx': droplet_map,
            'donor': [hto_names[i] for i in donor_indices]
        })
        gt = df_cells.groupby('droplet_idx')['donor'].apply(list).to_frame()
        gt['n_cells'] = gt['donor'].apply(len)
        gt['unique_donors'] = gt['donor'].apply(set).apply(len)

        # 4. 自动去除空液滴并重置索引
        active_indices = gt.index.values
        filtered_matrix = droplet_matrix[active_indices]
        final_gt = gt.reset_index(drop=True)

        return filtered_matrix, final_gt

    @staticmethod
    def display_report(gt):
        total = len(gt)
        singlets = gt[gt['n_cells'] == 1]
        msm = gt[(gt['n_cells'] > 1) & (gt['unique_donors'] > 1)]
        ssm = gt[(gt['n_cells'] > 1) & (gt['unique_donors'] == 1)]
        
        print(f"\n{'='*15} HTO Simulation QC Report {'='*15}")
        print(f"Non-Empty Droplets: {total}")
        print("-" * 50)
        print(f"Singlets:           {len(singlets):<8} | {len(singlets)/total:>7.2%}")
        print(f"MSM (Heterotypic):  {len(msm):<8} | {len(msm)/total:>7.2%}")
        print(f"SSM (Homotypic):    {len(ssm):<8} | {len(ssm)/total:>7.2%}")
        print("-" * 50)
        
        if len(singlets) > 0:
            print("Singlet Distribution (Top 5 HTOs):")
            print(singlets['donor'].apply(lambda x: x[0]).value_counts().head(5))
        print(f"{'='*50}\n")