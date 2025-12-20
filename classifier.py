import torch
import numpy as np
import pandas as pd
from typing import List, Optional

class HTClassifier:
    """Interprets model posteriors into final biological assignments."""
    
    def __init__(self, cfg_labels: List[str], tag_names: List[str], configs: torch.Tensor):
        self.cfg_labels = cfg_labels
        self.tag_names = tag_names
        self.configs = configs
        # 统计每个 configuration 激活了多少个通道
        self.cfg_sums = configs.sum(dim=1).cpu().numpy()

    def classify(self, posteriors: torch.Tensor, barcodes: np.ndarray, map_thresh: float = 0.9) -> pd.DataFrame:
        if posteriors is None: 
            raise ValueError("Posteriors not found. Run model first.")
        
        target_device = posteriors.device
        configs_on_device = self.configs.to(target_device)
        
        # 1. 获取最大后验索引和对应的概率值
        post_np = posteriors.detach().cpu().numpy()
        map_idx = np.argmax(post_np, axis=1)
        probs = post_np[np.arange(len(post_np)), map_idx]

        # 2. 翻译逻辑：不再单独区分 Negative，统一映射
        assignments_final = []
        assignments_raw_bio = [] # 记录生物学意义上的最像类别
        configs_cpu = self.configs.cpu().numpy()
        
        for i, idx in enumerate(map_idx):
            p = probs[i]
            s = self.cfg_sums[idx]
            
            # 确定生物学标签类型
            if s == 0:
                label = "Unassigned" # GMM 的全背景配置现在直接归为 Unassigned
            elif s == 1:
                tag_idx = int(np.where(configs_cpu[idx] == 1)[0][0])
                label = self.tag_names[tag_idx]
            else:
                label = "Multiplet"
            
            assignments_raw_bio.append(label)

            # 阈值过滤
            # 如果概率不够高，或者是背景类(s==0)，统一标记为 Unassigned
            if p < map_thresh or s == 0:
                assignments_final.append("Unassigned")
            else:
                assignments_final.append(label)

        # 3. 构建统一格式的 DataFrame
        df = pd.DataFrame({
            "barcode": barcodes,
            "assignment_raw": [self.cfg_labels[i] for i in map_idx], # 模型内部索引名
            "assignment": assignments_raw_bio, # 映射后的原始标签
            "assignment_final": assignments_final, # 经过阈值过滤的最终标签
            "probability": probs
        })

        # 4. 计算边际概率
        marginals = (posteriors @ configs_on_device).detach().cpu().numpy()
        for i, tag in enumerate(self.tag_names):
            df[f"p_{tag}"] = marginals[:, i]
            
        return df