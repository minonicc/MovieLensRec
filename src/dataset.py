import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MovieLensDataset(Dataset):
    """
    双塔召回数据集类：实现特征增强与长短期混合历史构造。
    """
    def __init__(self, csv_path, meta, neg_ratio=5, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.user_idx = self.df['user_idx'].values
        self.item_idx = self.df['item_idx'].values
        self.pos_in_seq = self.df['pos'].values 
        
        self.item_count = meta['item_count']
        self.user_sequences = meta['user_sequences']
        self.movie_genres = meta['movie_genres']
        self.item_feat_map = meta['item_feat_map']
        self.user_feat_map = meta['user_feat_map']
        
        self.neg_ratio = neg_ratio
        self.is_train = is_train
        self.max_hist_len = 60 # 50最近 + 10随机
        self.max_genre_len = 6
        self.max_hist_genre_len = 100 # 用户历史题材展平后的最大长度

    def _get_mixed_history(self, full_seq, pos):
        history_all = full_seq[:pos]
        if not history_all: return [0] * self.max_hist_len
        recent = history_all[-50:]
        rem = history_all[:-50]
        rand = random.sample(rem, 10) if len(rem) > 10 else rem
        combined = recent + rand
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

    def __len__(self): return len(self.user_idx)

    def __getitem__(self, idx):
        u_idx = self.user_idx[idx]
        target_i_idx = self.item_idx[idx]
        pos = self.pos_in_seq[idx]
        full_seq = self.user_sequences.get(u_idx, [])
        
        u_hist = self._get_mixed_history(full_seq, pos)
        u_hist_genres = self._get_hist_genres(u_hist)
        u_stats = self._get_user_feats(u_idx)
        
        if self.is_train:
            items = [target_i_idx]
            for _ in range(self.neg_ratio):
                neg_i = random.randint(1, self.item_count - 1)
                while neg_i in full_seq: neg_i = random.randint(1, self.item_count - 1)
                items.append(neg_i)
            
            i_genres = [self.movie_genres.get(i, [0])[:6] + [0]*(6-len(self.movie_genres.get(i, [0])[:6])) for i in items]
            i_stats = [self._get_item_feats(i) for i in items]
            labels = [1.0] + [0.0] * self.neg_ratio
            
            return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(u_hist_genres), torch.tensor(u_stats),
                    torch.tensor(items), torch.tensor(i_genres), torch.tensor(i_stats), torch.tensor(labels))
        else:
            i_genres = self.movie_genres.get(target_i_idx, [0])[:6] + [0]*(6-len(self.movie_genres.get(target_i_idx, [0])[:6]))
            i_stats = self._get_item_feats(target_i_idx)
            return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(u_hist_genres), torch.tensor(u_stats),
                    torch.tensor(target_i_idx), torch.tensor(i_genres), torch.tensor(i_stats))

def collate_fn(batch):
    res = [[] for _ in range(len(batch[0]))]
    for b in batch:
        # 如果是训练模式，第五个元素(items)是列表
        if b[4].dim() > 0: 
            for i in range(len(b[-1])): # 遍历 neg_ratio
                for k in range(4): res[k].append(b[k]) # User侧
                for k in range(4, 8): res[k].append(b[k][i]) # Item侧 + Label
        else:
            for k in range(len(b)): res[k].append(b[k])
    return [torch.stack(r) for r in res]
