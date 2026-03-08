import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RankingDataset(Dataset):
    """
    精排专用数据集加载器类。
    
    核心设计：
    1. 高性能读取：底层采用 Parquet 列式存储，相比 CSV 可大幅降低数据预处理与加载的 IO 耗时。
    2. 多类型特征支持：将特征解耦为 Categorical (离散)、Sequence (序列)、Numerical (数值) 三大板块。
    3. 特征对齐：确保特征名与模型 (model.py) 内部的输入 Key 严格对应。
    """
    def __init__(self, parquet_path):
        """
        :param parquet_path: 预处理好的精排样本文件路径
        """
        # 利用 pyarrow 引擎极速加载千万级排序样本
        self.df = pd.read_parquet(parquet_path)
        
        # --- 1. 离散类特征 (Categorical Features) ---
        # 这些特征将通过 Embedding 层映射为稠密向量。
        self.cat_cols = [
            'user_idx', 'item_idx', 'year_bucket', 
            'item_rating_bucket', 'item_count_bucket',
            'user_rating_bucket', 'user_count_bucket'
        ]
        
        # --- 2. 序列类特征 (Sequence Features) ---
        # 记录用户的动态历史轨迹，通常采用 Mean Pooling 后融入模型。
        self.seq_cols = ['hist_movie_ids', 'hist_genre_ids']
        
        # --- 3. 题材组合拳数值特征 (Numerical Features) ---
        # 这类特征通常不进行 Embedding 映射，而是通过归一化后直接拼接进入 MLP 的输入层。
        # 字段含义：题材浓度 (Density), 近期匹配 (Recent Match), 评分偏好 (Rating Bias)。
        self.num_cols = ['genre_density', 'genre_recent_match', 'genre_rating_bias']
        
        # 提取二分类目标标签 (Label=1 代表满意/转化，Label=0 代表负向反馈或未交互)
        self.labels = self.df['label'].values.astype(np.float32)

    def __len__(self):
        """
        返回数据集的总样本行数。
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        PyTorch 数据索引接口：将 DataFrame 的一行数据转化为 Tensor 字典。
        """
        row = self.df.iloc[idx]
        
        # A. 封装离散 ID 类特征 (LongTensor 类型，用于 Embedding 查找索引)
        cat_features = {col: torch.tensor(row[col], dtype=torch.long) for col in self.cat_cols}
        
        # B. 封装历史行为序列 (LongTensor 列表)
        seq_features = {col: torch.tensor(row[col], dtype=torch.long) for col in self.seq_cols}
        
        # C. 封装组合拳连续特征 (FloatTensor 类型，直接参与数值计算)
        num_features = {col: torch.tensor(row[col], dtype=torch.float32) for col in self.num_cols}
        
        # D. 提取标签张量
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return cat_features, seq_features, num_features, label

def ranking_collate_fn(batch):
    """
    精排数据批处理对齐函数。
    作用：将 Dataset 产生的单条样本字典，在 Batch 维度进行垂直堆叠，
    从而输出模型期望的 Batch Tensor 格式。
    """
    cat_features, seq_features, num_features = {}, {}, {}
    
    # 提取特征键名参考 (假设 Batch 内所有样本的键名一致)
    cat_keys = batch[0][0].keys()
    seq_keys = batch[0][1].keys()
    num_keys = batch[0][2].keys()
    
    # 批量堆叠离散 ID 矩阵
    for key in cat_keys:
        cat_features[key] = torch.stack([item[0][key] for item in batch])
        
    # 批量堆叠变长序列张量 (注意：此处已假设样本在 data_prep 阶段完成了 Padding)
    for key in seq_keys:
        seq_features[key] = torch.stack([item[1][key] for item in batch])
        
    # 批量堆叠连续数值向量
    for key in num_keys:
        num_features[key] = torch.stack([item[2][key] for item in batch])
        
    # 堆叠 Batch 级别的标签
    labels = torch.stack([item[3] for item in batch])
    
    return cat_features, seq_features, num_features, labels
