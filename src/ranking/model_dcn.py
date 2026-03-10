import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetworkV2(nn.Module):
    """
    DCNv2 核心组件：显式交叉网络 (Cross Network)。
    采用 Vector-wise 模式：特征以 Embedding 向量为单位进行全参数矩阵交叉。
    """
    def __init__(self, num_fields, embed_dim, num_layers=2):
        """
        :param num_fields: 特征块的数量 (本模型为 10)
        :param embed_dim: 每个特征块的维度 (16)
        :param num_layers: 交叉层的层数 (用户设定为 2)
        """
        super(CrossNetworkV2, self).__init__()
        self.num_layers = num_layers
        
        # 每一层交叉层都拥有独立的权重矩阵 W 和偏置 b
        # 权重形状为 [num_fields, num_fields]，实现特征块之间的全排列交互
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_fields, num_fields)) for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_fields, embed_dim)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        """
        前向计算公式：x_{l+1} = x0 * (W_l * x_l) + b_l + x_l
        :param x0: 原始输入特征矩阵 [Batch, Num_Fields, Embed_Dim]
        """
        xl = x0
        for i in range(self.num_layers):
            # 1. 线性变换：W_l * x_l -> [Batch, Num_Fields, Embed_Dim]
            # 这里利用 matmul 在 Field 维度进行加权组合
            lw = torch.matmul(self.cross_weights[i], xl) 
            
            # 2. 显式交叉：x0 与线性变换结果逐元素相乘，注入 x0 信息
            xl = x0 * (lw + self.cross_biases[i]) + xl
            
        return xl

class DCNv2(nn.Module):
    """
    精排模型：DCNv2 (Deep & Cross Network V2)。
    架构特点：并行 (Parallel) 结构，左侧为显式 Cross 层，右侧为标准 MLP 层。
    """
    def __init__(self, feature_dims, embed_dim=16, mlp_dims=[128, 64], cross_layers=2):
        super(DCNv2, self).__init__()
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        
        # --- 1. Embedding 层 (参数共享逻辑与 DeepFM 对齐) ---
        self.item_emb_layer = nn.Embedding(feature_dims['item_idx'], embed_dim, padding_idx=0)
        self.genre_emb_layer = nn.Embedding(feature_dims['hist_genre_ids'], embed_dim, padding_idx=0)
        self.user_emb_layer = nn.Embedding(feature_dims['user_idx'], embed_dim)
        
        self.stat_feat_names = [
            'year_bucket', 'item_rating_bucket', 'item_count_bucket', 
            'user_rating_bucket', 'user_count_bucket'
        ]
        self.stat_embs = nn.ModuleDict({
            name: nn.Embedding(feature_dims[name], embed_dim) for name in self.stat_feat_names
        })

        # --- 2. 数值特征投影层 ---
        # 将 3 个连续特征 (浓度、近期匹配、评分偏好) 投影到 16 维，参与 Cross 交叉
        self.num_projection = nn.Linear(3, embed_dim)

        # --- 3. 并行双支架定义 ---
        # A. Cross Network：输入为 10 个特征块 (9个离散 + 1个数值投影)
        num_fields = 9 + 1
        self.cross_network = CrossNetworkV2(num_fields, embed_dim, num_layers=cross_layers)
        
        # B. Deep MLP：结构与 DeepFM 保持一致以确保对比公平性
        input_dim = num_fields * embed_dim
        layers = []
        curr_dim = input_dim
        for next_dim in mlp_dims:
            layers.append(nn.Linear(curr_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            curr_dim = next_dim
        self.mlp = nn.Sequential(*layers)
        
        # 最终汇总层
        self.final_linear = nn.Linear(curr_dim + input_dim, 1)

    def _get_feature_matrix(self, cat_features, seq_features, num_features):
        """
        特征对齐辅助函数：将散装特征打包为统一格式的特征矩阵。
        返回形状：[Batch, 10, 16]
        """
        fields = []
        # 1. User & Item
        fields.append(self.user_emb_layer(cat_features['user_idx']))
        fields.append(self.item_emb_layer(cat_features['item_idx']))
        
        # 2. Stats
        for name in self.stat_feat_names:
            fields.append(self.stat_embs[name](cat_features[name]))
            
        # 3. Sequences (Mean Pooling)
        h_items = seq_features['hist_movie_ids']
        mask_i = (h_items != 0).float().unsqueeze(-1)
        fields.append(torch.sum(self.item_emb_layer(h_items) * mask_i, dim=1) / (torch.sum(mask_i, dim=1) + 1e-8))
        
        h_genres = seq_features['hist_genre_ids']
        mask_g = (h_genres != 0).float().unsqueeze(-1)
        fields.append(torch.sum(self.genre_emb_layer(h_genres) * mask_g, dim=1) / (torch.sum(mask_g, dim=1) + 1e-8))
        
        # 4. 数值投影
        num_vec = torch.stack([num_features['genre_density'], num_features['genre_recent_match'], num_features['genre_rating_bias']], dim=-1)
        fields.append(F.relu(self.num_projection(num_vec))) # 经过非线性激活投影至 16 维
        
        # 堆叠结果 [Batch, 10, 16]
        return torch.stack(fields, dim=1)

    def forward(self, cat_features, seq_features, num_features):
        """
        DCNv2 并行前向传播。
        """
        # 构造统一格式的特征矩阵
        x0 = self._get_feature_matrix(cat_features, seq_features, num_features)
        x_flat = x0.view(x0.size(0), -1) # 展平供 MLP 使用
        
        # 1. 运行 Cross Network 支路
        cross_out = self.cross_network(x0)
        cross_out_flat = cross_out.view(cross_out.size(0), -1)
        
        # 2. 运行 Deep MLP 支路
        deep_out = self.mlp(x_flat)
        
        # 3. 最终并行融合 (Concatenate)
        combined = torch.cat([cross_out_flat, deep_out], dim=-1)
        logits = self.final_linear(combined)
        
        return logits.squeeze(-1)
