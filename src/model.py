import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTowerModel(nn.Module):
    """
    增强版双塔召回模型：集成电影年份、均分、次数及用户偏好分布特征。
    当前版本：采用原始点积（Dot Product）计算匹配分。
    """
    def __init__(self, user_count, item_count, genre_count, embed_dim=64):
        super(DualTowerModel, self).__init__()
        
        # --- 共享物品 Embedding ---
        # item_id_embedding 既作为 Item 塔的输入，也作为 User 塔的历史序列输入。
        self.item_id_embedding = nn.Embedding(item_count, embed_dim, padding_idx=0)
        self.genre_embedding = nn.Embedding(genre_count, embed_dim, padding_idx=0)
        
        # --- 侧向特征 Embedding (分桶) ---
        # 统一使用 16 维的小 embedding 表达统计特征
        self.feat_embed_dim = 16
        self.year_emb = nn.Embedding(12, self.feat_embed_dim, padding_idx=0)
        self.rating_emb = nn.Embedding(12, self.feat_embed_dim, padding_idx=0)
        self.count_emb = nn.Embedding(12, self.feat_embed_dim, padding_idx=0)

        # --- User 塔层 ---
        self.user_id_embedding = nn.Embedding(user_count, embed_dim)
        # 输入：UserID(64) + HistSeq(64) + HistGenres(64) + UserRating(16) + UserCount(16) = 224
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + self.feat_embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # --- Item 塔层 ---
        # 输入：ItemID(64) + Genres(64) + Year(16) + ItemRating(16) + ItemCount(16) = 176
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + self.feat_embed_dim * 3, 128),
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

    def forward_user(self, user_indices, history_indices, hist_genres_indices, user_stats):
        """
        生成归一化后的用户塔向量（余弦相似度版本）。
        """
        # 1. 基础 Embedding
        u_emb = self.user_id_embedding(user_indices)
        h_emb = self._mean_pooling(self.item_id_embedding(history_indices), history_indices)
        hg_emb = self._mean_pooling(self.genre_embedding(hist_genres_indices), hist_genres_indices)
        
        # 2. 统计特征 (user_stats[:, 0] 是均分桶, [:, 1] 是次数桶)
        ur_emb = self.rating_emb(user_stats[:, 0])
        uc_emb = self.count_emb(user_stats[:, 1])
        
        combined = torch.cat([u_emb, h_emb, hg_emb, ur_emb, uc_emb], dim=-1)
        out = self.user_mlp(combined)
        
        # 核心改动：对输出向量进行 L2 归一化，使其模长为 1，配合温度系数实现余弦相似度
        return F.normalize(out, p=2, dim=-1)

    def forward_item(self, item_indices, genre_indices, item_stats):
        """
        生成归一化后的物品塔向量（余弦相似度版本）。
        """
        # 1. 基础 Embedding
        i_emb = self.item_id_embedding(item_indices)
        g_emb = self._mean_pooling(self.genre_embedding(genre_indices), genre_indices)
        
        # 2. 侧向特征 (item_stats[:, 0]是年份, [:, 1]是均分, [:, 2]是次数)
        iy_emb = self.year_emb(item_stats[:, 0])
        ir_emb = self.rating_emb(item_stats[:, 1])
        ic_emb = self.count_emb(item_stats[:, 2])
        
        combined = torch.cat([i_emb, g_emb, iy_emb, ir_emb, ic_emb], dim=-1)
        out = self.item_mlp(combined)
        
        # 核心改动：对输出向量进行 L2 归一化，使其模长为 1，配合温度系数实现余弦相似度
        return F.normalize(out, p=2, dim=-1)

    def forward(self, user_indices, history_indices, hist_genres, user_stats,
                item_indices, genre_indices, item_stats):
        """
        训练时的前向传播：计算用户向量与物品向量的归一化点积（余弦相似度）。
        """
        u_vec = self.forward_user(user_indices, history_indices, hist_genres, user_stats)
        i_vec = self.forward_item(item_indices, genre_indices, item_stats)
        
        # 归一化向量的内积即为余弦相似度，取值范围 [-1, 1]
        logits = torch.sum(u_vec * i_vec, dim=-1)
        # 注意：此处返回的是未经过温度系数缩放的相似度，缩放逻辑在 train.py 的 InfoNCE 中处理
        return torch.sigmoid(logits)
