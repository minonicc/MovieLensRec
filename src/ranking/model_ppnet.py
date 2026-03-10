import torch
import torch.nn as nn
import torch.nn.functional as F

class PPNetGate(nn.Module):
    """
    PPNet 门控单元。
    """
    def __init__(self, input_dim, output_dim):
        super(PPNetGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, u_info):
        # 双向缩放 (0, 2)
        return self.gate(u_info) * 2

class CrossNetworkV2(nn.Module):
    """
    DCNv2 交叉网络。
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

class DCN_PPNet(nn.Module):
    """
    精排模型：DCNv2 + PPNet 个性化增强。
    """
    def __init__(self, feature_dims, embed_dim=16, mlp_dims=[128, 64], cross_layers=2):
        super(DCN_PPNet, self).__init__()
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        
        # --- 1. Embedding 层 ---
        self.item_emb_layer = nn.Embedding(feature_dims['item_idx'], embed_dim, padding_idx=0)
        self.genre_emb_layer = nn.Embedding(feature_dims['hist_genre_ids'], embed_dim, padding_idx=0)
        self.user_emb_layer = nn.Embedding(feature_dims['user_idx'], embed_dim)
        
        self.stat_feat_names = ['year_bucket', 'item_rating_bucket', 'item_count_bucket', 
                                'user_rating_bucket', 'user_count_bucket']
        self.stat_embs = nn.ModuleDict({
            name: nn.Embedding(feature_dims[name], embed_dim) for name in self.stat_feat_names
        })

        # --- 2. 数值投影层 ---
        self.num_projection = nn.Linear(3, embed_dim)

        # --- 3. 支路定义 ---
        num_fields = 9 + 1
        # Cross 支路
        self.cross_network = CrossNetworkV2(num_fields, embed_dim, num_layers=cross_layers)
        
        # Deep 支路
        input_dim = num_fields * embed_dim
        self.mlp_layer1 = nn.Sequential(
            nn.Linear(input_dim, mlp_dims[0]),
            nn.BatchNorm1d(mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.mlp_layer2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        
        # --- 4. PPNet 门控单元 ---
        # 门控输入维度：User ID (16) + User_Rating (16) + User_Count (16) = 48 维
        gate_input_dim = embed_dim * 3
        self.ppnet_gate = PPNetGate(gate_input_dim, mlp_dims[1])
        
        # 最终汇总
        self.final_linear = nn.Linear(mlp_dims[1] + input_dim, 1)

    def _get_user_info_vector(self, cat_features):
        """
        提取用户强个性化特征，并实施梯度截断保护。
        """
        u_emb = self.user_emb_layer(cat_features['user_idx'])
        ur_emb = self.stat_embs['user_rating_bucket'](cat_features['user_rating_bucket'])
        uc_emb = self.stat_embs['user_count_bucket'](cat_features['user_count_bucket'])
        
        # 组合用户特征
        combined = torch.cat([u_emb, ur_emb, uc_emb], dim=-1)
        
        # 核心：使用 detach() 截断梯度。
        # 此举防止门控支路的剧烈梯度回传破坏底层用户 Embedding 权重的稳健更新。
        return combined.detach()

    def _get_feature_matrix(self, cat_features, seq_features, num_features):
        """
        构造标准特征矩阵 [Batch, 10, 16]。
        注意：此处产生的 Embedding 会通过主支路进行正常的梯度更新。
        """
        fields = []
        fields.append(self.user_emb_layer(cat_features['user_idx']))
        fields.append(self.item_emb_layer(cat_features['item_idx']))
        for name in self.stat_feat_names:
            fields.append(self.stat_embs[name](cat_features[name]))
        
        # 序列处理
        h_it = seq_features['hist_movie_ids']; mask_i = (h_it != 0).float().unsqueeze(-1)
        fields.append(torch.sum(self.item_emb_layer(h_it)*mask_i, dim=1)/(torch.sum(mask_i, dim=1)+1e-8))
        h_gs = seq_features['hist_genre_ids']; mask_g = (h_gs != 0).float().unsqueeze(-1)
        fields.append(torch.sum(self.genre_emb_layer(h_gs)*mask_g, dim=1)/(torch.sum(mask_g, dim=1)+1e-8))
        
        # 数值投影
        num_v = torch.stack([num_features['genre_density'], num_features['genre_recent_match'], num_features['genre_rating_bias']], dim=-1)
        fields.append(F.relu(self.num_projection(num_v)))
        
        return torch.stack(fields, dim=1)

    def forward(self, cat_features, seq_features, num_features):
        # 1. 特征底座
        x0 = self._get_feature_matrix(cat_features, seq_features, num_features)
        x_flat = x0.view(x0.size(0), -1)
        
        # 2. Cross 网络 (全局逻辑)
        cross_out = self.cross_network(x0).view(x0.size(0), -1)
        
        # 3. Deep 网络 (个性化门控干预)
        d_out = self.mlp_layer1(x_flat)
        d_out = self.mlp_layer2(d_out)
        
        # --- PPNet 门控干预 ---
        # 此时获取的 u_info 是梯度截断的，保护了底层参数
        u_info = self._get_user_info_vector(cat_features)
        gate_v = self.ppnet_gate(u_info)
        d_out = d_out * gate_v
        d_out = F.relu(d_out)
        
        # 4. 融合输出
        combined = torch.cat([cross_out, d_out], dim=-1)
        return self.final_linear(combined).squeeze(-1)
