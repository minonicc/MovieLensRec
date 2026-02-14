import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTowerModel(nn.Module):
    def __init__(self, user_count, item_count, genre_count, embed_dim=64):
        super(DualTowerModel, self).__init__()
        
        # --- User 塔 ---
        self.user_id_embedding = nn.Embedding(user_count, embed_dim)
        # 替换 EmbeddingBag 为标准 Embedding
        self.item_emb_for_user = nn.Embedding(item_count + 1, embed_dim, padding_idx=0)
        
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # --- Item 塔 ---
        self.item_id_embedding = nn.Embedding(item_count, embed_dim)
        self.genre_embedding = nn.Embedding(genre_count + 1, embed_dim, padding_idx=0)
        
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def _mean_pooling(self, embeddings, indices):
        # 手动实现 Mean Pooling，忽略 padding (0)
        mask = (indices != 0).float().unsqueeze(-1) # [B, SeqLen, 1]
        sum_embeddings = torch.sum(embeddings * mask, dim=1) # [B, Dim]
        count = torch.sum(mask, dim=1) + 1e-8 # [B, 1]
        return sum_embeddings / count

    def forward_user(self, user_indices, history_indices):
        u_emb = self.user_id_embedding(user_indices)
        h_embs = self.item_emb_for_user(history_indices)
        h_emb = self._mean_pooling(h_embs, history_indices)
        
        combined = torch.cat([u_emb, h_emb], dim=-1)
        return self.user_mlp(combined)

    def forward_item(self, item_indices, genre_indices):
        i_emb = self.item_id_embedding(item_indices)
        g_embs = self.genre_embedding(genre_indices)
        g_emb = self._mean_pooling(g_embs, genre_indices)
        
        combined = torch.cat([i_emb, g_emb], dim=-1)
        return self.item_mlp(combined)

    def forward(self, user_indices, history_indices, item_indices, genre_indices):
        u_vec = self.forward_user(user_indices, history_indices)
        i_vec = self.forward_item(item_indices, genre_indices)
        
        # 内积相似度
        logits = torch.sum(u_vec * i_vec, dim=-1)
        return torch.sigmoid(logits)
