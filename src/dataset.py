import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MovieLensDataset(Dataset):
    def __init__(self, csv_path, meta, neg_ratio=5, is_train=True, max_hist_len=50):
        self.data = pd.read_csv(csv_path)
        self.user_idx = self.data['user_idx'].values
        self.item_idx = self.data['item_idx'].values
        self.item_count = meta['item_count']
        self.user_history = meta['user_history']
        self.movie_genres = meta['movie_genres']
        self.neg_ratio = neg_ratio
        self.is_train = is_train
        self.max_hist_len = max_hist_len
        self.max_genre_len = 6 # 电影类型通常很少，6个足够

    def _get_padded_list(self, raw_list, max_len):
        # 截断或填充
        if len(raw_list) > max_len:
            return raw_list[-max_len:]
        else:
            return raw_list + [0] * (max_len - len(raw_list))

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        u_idx = self.user_idx[idx]
        pos_i_idx = self.item_idx[idx]
        
        # 准备用户历史 (已填充)
        u_hist = self._get_padded_list(self.user_history.get(u_idx, []), self.max_hist_len)
        
        if self.is_train:
            items = [pos_i_idx]
            labels = [1.0]
            for _ in range(self.neg_ratio):
                neg_i_idx = random.randint(0, self.item_count - 1)
                while neg_i_idx in self.user_history.get(u_idx, []):
                    neg_i_idx = random.randint(0, self.item_count - 1)
                items.append(neg_i_idx)
                labels.append(0.0)
            
            # 准备每个 item 的 genres (已填充)
            item_genres = [self._get_padded_list(self.movie_genres.get(i, []), self.max_genre_len) for i in items]
            
            return (torch.tensor(u_idx), 
                    torch.tensor(u_hist), 
                    torch.tensor(items), 
                    torch.tensor(item_genres), 
                    torch.tensor(labels))
        else:
            item_genres = self._get_padded_list(self.movie_genres.get(pos_i_idx, []), self.max_genre_len)
            return (torch.tensor(u_idx), 
                    torch.tensor(u_hist), 
                    torch.tensor(pos_i_idx),
                    torch.tensor(item_genres))

def collate_fn(batch):
    # 针对 Pointwise 展开 batch
    u_indices = []
    u_hists = []
    i_indices = []
    i_genres = []
    labels = []
    
    for b in batch:
        u_idx, u_hist, items, item_genres, lbls = b
        # items 是 (1 + neg_ratio) 长度
        for i in range(len(items)):
            u_indices.append(u_idx)
            u_hists.append(u_hist)
            i_indices.append(items[i])
            i_genres.append(item_genres[i])
            labels.append(lbls[i])
            
    return (torch.stack(u_indices), 
            torch.stack(u_hists), 
            torch.stack(i_indices), 
            torch.stack(i_genres), 
            torch.stack(labels))
