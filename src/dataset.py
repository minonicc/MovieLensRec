import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MovieLensDataset(Dataset):
    """
    双塔召回数据集类：实现特征增强与长短期混合历史构造。
    当前版本：仅返回正样本特征，负采样由训练循环统一处理以提升性能。
    """
    def __init__(self, csv_path, meta, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.user_idx = self.df['user_idx'].values
        self.item_idx = self.df['item_idx'].values
        self.pos_in_seq = self.df['pos'].values 
        
        self.item_count = meta['item_count']
        self.user_sequences = meta['user_sequences']
        self.movie_genres = meta['movie_genres']
        self.item_feat_map = meta['item_feat_map']
        self.user_feat_map = meta['user_feat_map']
        
        self.is_train = is_train
        self.max_hist_len = 60 # 50最近 + 10随机
        self.max_genre_len = 6
        self.max_hist_genre_len = 100 # 用户历史题材展平后的最大长度

    def _get_mixed_history(self, full_seq, pos):
        """
        核心逻辑：从 pos 之前的序列中，提取 50个最近物品 和 10个随机物品。
        """
        history_all = full_seq[:pos]
        if not history_all: return [0] * self.max_hist_len
        recent = history_all[-50:]
        rem = history_all[:-50]
        rand = random.sample(rem, 10) if len(rem) > 10 else rem
        combined = recent + rand
        # 补齐到 60
        return combined + [0] * (self.max_hist_len - len(combined))

    def _get_item_feats(self, i_idx):
        # 提取电影的侧向特征桶 ID
        f = self.item_feat_map.get(i_idx, {})
        return [f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)]

    def _get_user_feats(self, u_idx):
        # 提取用户的侧向特征桶 ID
        f = self.user_feat_map.get(u_idx, {})
        return [f.get('rating_bucket', 0), f.get('count_bucket', 0)]

    def _get_hist_genres(self, hist_ids):
        # 获取用户历史序列中所有电影的题材，并展平
        genres = []
        for i in hist_ids:
            if i == 0: continue
            genres.extend(self.movie_genres.get(i, []))
        if len(genres) > self.max_hist_genre_len: genres = genres[:self.max_hist_genre_len]
        return genres + [0] * (self.max_hist_genre_len - len(genres))

    def _get_padded_genres(self, i_idx):
        # 获取电影类型列表，并 Padding 到固定长度（6）。如果电影没有类型，则全为 0。
        g = self.movie_genres.get(i_idx, [])
        if len(g) > self.max_genre_len: return g[:self.max_genre_len]
        return g + [0] * (self.max_genre_len - len(g))

    def __len__(self): return len(self.user_idx)

    def __getitem__(self, idx):
        u_idx = self.user_idx[idx]
        target_i_idx = self.item_idx[idx]
        pos = self.pos_in_seq[idx]
        full_seq = self.user_sequences.get(u_idx, [])
        
        # 构造用户侧特征
        u_hist = self._get_mixed_history(full_seq, pos)
        u_hgenres = self._get_hist_genres(u_hist)
        u_stats = self._get_user_feats(u_idx)
        
        # 构造物品侧特征 (仅正样本)
        i_genres = self._get_padded_genres(target_i_idx)
        i_stats = self._get_item_feats(target_i_idx)

        # 仅返回正样本对的相关特征
        return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(u_hgenres), torch.tensor(u_stats),
                torch.tensor(target_i_idx), torch.tensor(i_genres), torch.tensor(i_stats))

def train_collate_fn(batch):
    """
    精简版 collate_fn：直接堆叠正样本张量。
    """
    return [torch.stack(r) for r in zip(*batch)]
