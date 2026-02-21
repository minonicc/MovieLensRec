import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTowerModel(nn.Module):
    """
    双塔召回模型 (Dual-Tower / DSSM 变体)。
    实现了 User 塔和 Item 塔的参数共享（用户历史点击 Embedding 复用物品 Embedding）。
    """
    def __init__(self, user_count, item_count, genre_count, embed_dim=64):
        super(DualTowerModel, self).__init__()
        
        # --- 核心：权重共享的 Embedding 层 ---
        # item_id_embedding 既作为 Item 塔的输入，也作为 User 塔的历史序列输入。
        self.item_id_embedding = nn.Embedding(item_count, embed_dim, padding_idx=0)
        
        # --- User 塔层 ---
        self.user_id_embedding = nn.Embedding(user_count, embed_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # --- Item 塔层 ---
        # 电影类型 Embedding
        self.genre_embedding = nn.Embedding(genre_count, embed_dim, padding_idx=0)
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def _mean_pooling(self, embeddings, indices):
        """
        手动实现 Mean Pooling (均值池化)，并排除 padding (索引为0) 的影响。
        """
        mask = (indices != 0).float().unsqueeze(-1) # [Batch, SeqLen, 1]
        sum_embeddings = torch.sum(embeddings * mask, dim=1) # [Batch, Dim]
        count = torch.sum(mask, dim=1) + 1e-8 # 避免除以0
        return sum_embeddings / count

    def forward_user(self, user_indices, history_indices):
        """
        生成用户塔向量。
        """
        # 1. 查询用户 ID 向量
        u_emb = self.user_id_embedding(user_indices)
        # 2. 查询用户历史点击序列的向量 (这里实现了权重共享)
        h_embs = self.item_id_embedding(history_indices)
        # 3. 对历史序列进行均值聚合
        h_emb = self._mean_pooling(h_embs, history_indices)
        
        # 4. 拼接并通过 MLP
        combined = torch.cat([u_emb, h_emb], dim=-1)
        return self.user_mlp(combined)

    def forward_item(self, item_indices, genre_indices):
        """
        生成物品塔向量。
        """
        # 1. 查询电影 ID 向量
        i_emb = self.item_id_embedding(item_indices)
        # 2. 查询电影类型向量并聚合
        g_embs = self.genre_embedding(genre_indices)
        g_emb = self._mean_pooling(g_embs, genre_indices)
        
        # 3. 拼接并通过 MLP
        combined = torch.cat([i_emb, g_emb], dim=-1)
        return self.item_mlp(combined)

    def forward(self, user_indices, history_indices, item_indices, genre_indices):
        """
        训练时的前向传播：计算用户向量与物品向量的内积得分。
        """
        u_vec = self.forward_user(user_indices, history_indices)
        i_vec = self.forward_item(item_indices, genre_indices)
        
        # 向量点积代表相似度
        logits = torch.sum(u_vec * i_vec, dim=-1)
        # Pointwise 模式使用 Sigmoid 转化为概率
        return torch.sigmoid(logits)
