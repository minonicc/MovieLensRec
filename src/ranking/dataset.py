import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RankingDataset(Dataset):
    """
    精排专用数据集：支持题材组合拳特征的读取。
    用于 DeepFM / DCNv2 等 Pointwise 模型训练。
    """
    def __init__(self, parquet_path):
        # 使用高性能 pyarrow 引擎加载数据
        self.df = pd.read_parquet(parquet_path)
        
        # 1. 离散类特征 (Categorical)
        self.cat_cols = [
            'user_idx', 'item_idx', 'year_bucket', 
            'item_rating_bucket', 'item_count_bucket',
            'user_rating_bucket', 'user_count_bucket'
        ]
        
        # 2. 序列类特征 (Sequence)
        self.seq_cols = ['hist_movie_ids', 'hist_genre_ids']
        
        # 3. 题材组合拳数值特征 (Numerical)
        self.num_cols = ['genre_density', 'genre_recent_match', 'genre_rating_bias']
        
        # 提取目标标签
        self.labels = self.df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 封装离散特征
        cat_features = {col: torch.tensor(row[col], dtype=torch.long) for col in self.cat_cols}
        
        # 封装序列特征
        seq_features = {col: torch.tensor(row[col], dtype=torch.long) for col in self.seq_cols}
        
        # 封装组合拳连续特征
        num_features = {col: torch.tensor(row[col], dtype=torch.float32) for col in self.num_cols}
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return cat_features, seq_features, num_features, label

def ranking_collate_fn(batch):
    """
    批处理整理函数：将单个样本字典堆叠为 Batch 张量。
    """
    cat_features, seq_features, num_features = {}, {}, {}
    
    # 提取键名参考
    cat_keys = batch[0][0].keys()
    seq_keys = batch[0][1].keys()
    num_keys = batch[0][2].keys()
    
    for key in cat_keys:
        cat_features[key] = torch.stack([item[0][key] for item in batch])
        
    for key in seq_keys:
        seq_features[key] = torch.stack([item[1][key] for item in batch])
        
    for key in num_keys:
        num_features[key] = torch.stack([item[2][key] for item in batch])
        
    labels = torch.stack([item[3] for item in batch])
    
    return cat_features, seq_features, num_features, labels
