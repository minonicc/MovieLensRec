import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MovieLensDataset(Dataset):
    """
    双塔召回数据集类：实现『50最近+10随机』特征构造。
    """
    def __init__(self, csv_path, meta, neg_ratio=5, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.user_idx = self.df['user_idx'].values
        self.item_idx = self.df['item_idx'].values
        self.pos_in_seq = self.df['pos'].values 
        
        self.item_count = meta['item_count']
        self.user_sequences = meta['user_sequences']
        self.movie_genres = meta['movie_genres']
        
        self.neg_ratio = neg_ratio
        self.is_train = is_train
        
        # --- 特征宽度定义 ---
        self.recent_len = 50
        self.random_len = 10
        self.max_hist_len = self.recent_len + self.random_len # 总宽度 60
        self.max_genre_len = 6

    def _get_mixed_history(self, full_seq, pos):
        """
        核心逻辑：从 pos 之前的序列中，提取 50个最近物品 和 10个随机物品。
        """
        history_all = full_seq[:pos]
        if not history_all:
            return [0] * self.max_hist_len
        
        # 1. 提取最近 50 个
        recent_part = history_all[-self.recent_len:]
        
        # 2. 提取远期随机 10 个 (从排除掉最近 50 个之后的剩余部分选)
        remaining_part = history_all[:-self.recent_len]
        if len(remaining_part) > self.random_len:
            random_part = random.sample(remaining_part, self.random_len)
        else:
            # 如果剩余不足 10 个，则全取
            random_part = remaining_part
            
        # 3. 合并并 Padding 到 60
        combined = recent_part + random_part
        if len(combined) < self.max_hist_len:
            combined = combined + [0] * (self.max_hist_len - len(combined))
        return combined

    def _get_padded_genres(self, i_idx):
        g = self.movie_genres.get(i_idx, [])
        if len(g) > self.max_genre_len: return g[:self.max_genre_len]
        return g + [0] * (self.max_genre_len - len(g))

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        u_idx = self.user_idx[idx]
        target_i_idx = self.item_idx[idx]
        pos = self.pos_in_seq[idx]
        full_seq = self.user_sequences.get(u_idx, [])
        
        # 获取混合历史特征 (50最近 + 10随机)
        u_hist = self._get_mixed_history(full_seq, pos)
        
        if self.is_train:
            items = [target_i_idx]
            labels = [1.0]
            # 负采样
            for _ in range(self.neg_ratio):
                neg_i = random.randint(1, self.item_count - 1)
                while neg_i in full_seq: # 严禁负样本出现在用户历史中
                    neg_i = random.randint(1, self.item_count - 1)
                items.append(neg_i)
                labels.append(0.0)
            
            item_genres = [self._get_padded_genres(i) for i in items]
            return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(items), torch.tensor(item_genres), torch.tensor(labels))
        else:
            item_genres = self._get_padded_genres(target_i_idx)
            return (torch.tensor(u_idx), torch.tensor(u_hist), torch.tensor(target_i_idx), torch.tensor(item_genres))

def collate_fn(batch):
    u_indices, u_hists, i_indices, i_genres, labels = [], [], [], [], []
    for b in batch:
        u_idx, u_hist, items, genres, lbls = b
        for i in range(len(items)):
            u_indices.append(u_idx); u_hists.append(u_hist)
            i_indices.append(items[i]); i_genres.append(genres[i]); labels.append(lbls[i])
    return torch.stack(u_indices), torch.stack(u_hists), torch.stack(i_indices), torch.stack(i_genres), torch.stack(labels)
