import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MovieLensDataset(Dataset):
    """
    双塔召回数据集类：实现多特征增强、长短期混合历史构造以及专属困难负样本的精准加载。
    """
    def __init__(self, csv_path, meta, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.user_idx = self.df['user_idx'].values
        self.item_idx = self.df['item_idx'].values
        self.pos_in_seq = self.df['pos'].values 
        
        # 在训练模式下，显式加载预处理好的专属困难负样本 ID
        self.is_train = is_train
        if self.is_train:
            self.hard_neg_idx = self.df['hard_neg_idx'].values

        self.item_count = meta['item_count']
        self.user_sequences = meta['user_sequences'] # 用户全量点击序列，用于动态截取特征
        self.movie_genres = meta['movie_genres']     # 电影 ID 映射到题材列表
        self.item_feat_map = meta['item_feat_map']   # 电影侧向特征桶映射
        self.user_feat_map = meta['user_feat_map']   # 用户侧向特征桶映射
        
        # 特征宽度定义
        self.max_hist_len = 60 # 包含 50个最近 + 10个远期随机物品
        self.max_genre_len = 6 # 电影类型的最大截断长度
        self.max_hist_genre_len = 100 # 用户历史题材展平后的最大聚合长度

    def _get_mixed_history(self, full_seq, pos):
        """
        混合历史特征构造逻辑：从当前位置 pos 之前的序列中，提取最近的 50 个物品和远期随机的 10 个物品。
        这种『50+10』的非对称设计旨在同时捕捉用户当前的即时兴趣和底层的本质偏好。
        """
        history_all = full_seq[:pos]
        if not history_all: return [0] * self.max_hist_len
        
        # 1. 提取最近 50 个物品
        recent_part = history_all[-50:]
        # 2. 从剩余历史中随机选择 10 个物品作为长期兴趣『锚点』
        remaining_part = history_all[:-50]
        if len(remaining_part) > 10:
            random_part = random.sample(remaining_part, 10)
        else:
            random_part = remaining_part
            
        combined = recent_part + random_part
        # 补齐到固定长度 60
        return combined + [0] * (self.max_hist_len - len(combined))

    def _get_item_feats(self, i_idx):
        """
        从预处理的快照中提取电影的侧向特征桶 ID (年份, 均分, 次数)。
        """
        f = self.item_feat_map.get(i_idx, {})
        return [f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)]

    def _get_user_feats(self, u_idx):
        """
        从预处理的快照中提取用户的侧向特征桶 ID (均分, 次数)。
        """
        f = self.user_feat_map.get(u_idx, {})
        return [f.get('rating_bucket', 0), f.get('count_bucket', 0)]

    def _get_hist_genres(self, hist_ids):
        """
        将用户历史序列中所有电影涉及到的题材进行展平聚合，形成偏好分布。
        """
        genres = []
        for i in hist_ids:
            if i == 0: continue
            genres.extend(self.movie_genres.get(i, []))
        if len(genres) > self.max_hist_genre_len: genres = genres[:self.max_hist_genre_len]
        return genres + [0] * (self.max_hist_genre_len - len(genres))

    def _get_padded_genres(self, i_idx):
        """
        获取电影的题材列表并补齐至固定长度 6。
        """
        g = self.movie_genres.get(i_idx, [])
        if len(g) > self.max_genre_len: return g[:self.max_genre_len]
        return g + [0] * (self.max_genre_len - len(g))

    def __len__(self): return len(self.user_idx)

    def __getitem__(self, idx):
        u_idx = self.user_idx[idx]
        target_i_idx = self.item_idx[idx]
        pos = self.pos_in_seq[idx]
        full_seq = self.user_sequences.get(u_idx, [])
        
        # 构造这一时刻的用户侧综合特征
        u_hist = self._get_mixed_history(full_seq, pos)
        u_hgenres = self._get_hist_genres(u_hist)
        u_stats = self._get_user_feats(u_idx)
        
        # 构造当前正样本物品的侧向特征
        i_genres = self._get_padded_genres(target_i_idx)
        i_stats = self._get_item_feats(target_i_idx)

        if self.is_train:
            # 训练模式：除了正样本，同步加载预分配好的专属困难负样本及其特征
            h_idx = self.hard_neg_idx[idx]
            h_genres = self._get_padded_genres(h_idx)
            h_stats = self._get_item_feats(h_idx)
            
            return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(u_hgenres), torch.tensor(u_stats),
                    torch.tensor(target_i_idx), torch.tensor(i_genres), torch.tensor(i_stats),
                    torch.tensor(h_idx), torch.tensor(h_genres), torch.tensor(h_stats))
        else:
            # 验证/测试模式：仅返回用户侧特征和正样本物品侧特征
            return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(u_hgenres), torch.tensor(u_stats),
                    torch.tensor(target_i_idx), torch.tensor(i_genres), torch.tensor(i_stats))

def train_collate_fn(batch):
    """
    批处理整理函数：支持 InfoNCE 混合负采样下的多特征对齐。
    """
    return [torch.stack(r) for r in zip(*batch)]
