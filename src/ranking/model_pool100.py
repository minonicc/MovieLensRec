import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetworkV2(nn.Module):
    """
    DCNv2 交叉网络组件。
    """
    def __init__(self, num_fields, embed_dim, num_layers=2):
        super(CrossNetworkV2, self).__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_fields, num_fields)) for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_fields, embed_dim)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            lw = torch.matmul(self.cross_weights[i], xl) 
            xl = x0 * (lw + self.cross_biases[i]) + xl
        return xl

class DCN_Pool100(nn.Module):
    """
    精排模型：DCNv2 + PPNet + 100维 Mean Pooling。
    本模型作为 DIN 的基准对比版本，验证 100 维长序列在无注意力机制下的性能表现。
    """
    def __init__(self, feature_dims, embed_dim=16, mlp_dims=[128, 64], cross_layers=2):
        super(DCN_Pool100, self).__init__()
        self.embed_dim = embed_dim
        
        # --- 1. Embedding 层 (参数共享) ---
        self.item_emb_layer = nn.Embedding(feature_dims['item_idx'], embed_dim, padding_idx=0)
        self.genre_emb_layer = nn.Embedding(feature_dims['hist_genre_ids'], embed_dim, padding_idx=0)
        self.user_emb_layer = nn.Embedding(feature_dims['user_idx'], embed_dim)
        
        self.stat_feat_names = ['year_bucket', 'item_rating_bucket', 'item_count_bucket', 
                                'user_rating_bucket', 'user_count_bucket']
        self.stat_embs = nn.ModuleDict({
            name: nn.Embedding(feature_dims[name], embed_dim) for name in self.stat_feat_names
        })

        # --- 2. 数值特征投影 ---
        self.num_projection = nn.Linear(3, embed_dim)

        # --- 3. 核心支路定义 ---
        # 保持 10 个特征块 (User, Item, Pooled_ID, Pooled_Genre, 5*Stats, 1*NumProjection)
        num_fields = 9 + 1
        self.cross_network = CrossNetworkV2(num_fields, embed_dim, num_layers=cross_layers)
        
        input_dim = num_fields * embed_dim
        self.mlp_layer1 = nn.Sequential(
            nn.Linear(input_dim, mlp_dims[0]),
            nn.BatchNorm1d(mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.mlp_layer2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        
        # --- 4. PPNet 门控 ---
        self.ppnet_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, mlp_dims[1]),
            nn.Sigmoid()
        )
        
        self.final_linear = nn.Linear(mlp_dims[1] + input_dim, 1)

    def forward(self, cat_features, seq_features, num_features):
        # A. 提取基础 Embedding
        u_emb = self.user_emb_layer(cat_features['user_idx'])
        i_emb = self.item_emb_layer(cat_features['item_idx'])
        
        # --- B. 执行 100 维 ID 序列的静态 Mean Pooling ---
        h_items = seq_features['hist_movie_ids']
        mask_i = (h_items != 0).float().unsqueeze(-1)
        h_item_embs = self.item_emb_layer(h_items)
        id_pooled_emb = torch.sum(h_item_embs * mask_i, dim=1) / (torch.sum(mask_i, dim=1) + 1e-8)
        
        # C. 执行 100 维题材序列的静态 Mean Pooling
        h_genres = seq_features['hist_genre_ids']
        mask_g = (h_genres != 0).float().unsqueeze(-1)
        genre_pooled_emb = torch.sum(self.genre_emb_layer(h_genres) * mask_g, dim=1) / (torch.sum(mask_g, dim=1) + 1e-8)
        
        # D. 构造特征矩阵 [Batch, 10, 16]
        fields = [u_emb, i_emb, id_pooled_emb, genre_pooled_emb]
        for name in self.stat_feat_names:
            fields.append(self.stat_embs[name](cat_features[name]))
        
        num_v = torch.stack([num_features['genre_density'], num_features['genre_recent_match'], num_features['genre_rating_bias']], dim=-1)
        fields.append(F.relu(self.num_projection(num_v)))
        
        x0 = torch.stack(fields, dim=1)
        x_flat = x0.view(x0.size(0), -1)
        
        # 1. Cross 支路
        cross_out = self.cross_network(x0).view(x0.size(0), -1)
        
        # 2. Deep 支路
        d_out = self.mlp_layer1(x_flat)
        d_out = self.mlp_layer2(d_out)
        
        # PPNet 门控
        u_info = torch.cat([u_emb, self.stat_embs['user_rating_bucket'](cat_features['user_rating_bucket']), 
                            self.stat_embs['user_count_bucket'](cat_features['user_count_bucket'])], dim=-1).detach()
        gate_v = self.ppnet_gate(u_info) * 2
        d_out = F.relu(d_out * gate_v)
        
        # 3. 融合
        combined = torch.cat([cross_out, d_out], dim=-1)
        return self.final_linear(combined).squeeze(-1)
