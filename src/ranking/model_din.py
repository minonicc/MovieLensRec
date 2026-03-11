import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionUnit(nn.Module):
    """
    DIN 核心组件：Target Attention 单元。
    利用 PyTorch 矩阵广播机制实现 100 个历史物品的并行权重计算。
    """
    def __init__(self, embed_dim):
        super(AttentionUnit, self).__init__()
        # 交互向量包含：[Target, History, Target-History, Target*History] 共 4 倍维度
        self.fc = nn.Sequential(
            nn.Linear(4 * embed_dim, 32),
            nn.PReLU(), # 使用 PReLU 加速收敛且保持稳定性
            nn.Linear(32, 1)
        )

    def forward(self, target_emb, history_embs, mask):
        """
        :param target_emb: 候选物品向量 [Batch, Dim]
        :param history_embs: 历史序列向量 [Batch, SeqLen, Dim]
        :param mask: Padding 掩码 [Batch, SeqLen]
        """
        seq_len = history_embs.size(1)
        # 1. 广播扩展候选物品向量：[Batch, 1, Dim] -> [Batch, SeqLen, Dim]
        target_emb_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 2. 构造交互特征张量
        # 拼接：原始特征、差异特征、内积特征
        combined = torch.cat([
            target_emb_expanded, 
            history_embs, 
            target_emb_expanded - history_embs, 
            target_emb_expanded * history_embs
        ], dim=-1) # [Batch, SeqLen, 4*Dim]
        
        # 3. 并行计算 100 个位置的注意力原始得分
        # nn.Linear 会自动作用在最后一维
        scores = self.fc(combined).squeeze(-1) # [Batch, SeqLen]
        
        # 4. 应用 Padding Mask：将 Padding 位置的得分设为极小值，消除其对求和的影响
        paddings = torch.ones_like(scores) * (-2**32 + 1)
        scores = torch.where(mask == 0, paddings, scores)
        
        # 5. 计算注意力权重 (遵循 DIN 论文，不强求 Softmax 归一化以保留行为强度)
        # 但为了数值稳定，通常做一层轻量的缩放或直接 Sum Pooling
        # 此处我们直接使用加权求和
        att_weights = torch.sigmoid(scores).unsqueeze(-1) # [Batch, SeqLen, 1]
        
        # 6. 加权聚合：[Batch, SeqLen, 1] * [Batch, SeqLen, Dim] -> [Batch, Dim]
        pooled_emb = torch.sum(att_weights * history_embs, dim=1)
        
        return pooled_emb

class CrossNetworkV2(nn.Module):
    """
    DCNv2 交叉网络 (Vector-wise)。
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

class DIN_DCN_PPNet(nn.Module):
    """
    终极精排模型：DIN (ID 注意力) + DCNv2 (高阶交叉) + PPNet (个性化门控)。
    """
    def __init__(self, feature_dims, embed_dim=16, mlp_dims=[128, 64], cross_layers=2):
        super(DIN_DCN_PPNet, self).__init__()
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

        # --- 3. DIN 注意力单元 (仅针对 ID) ---
        self.attention_unit = AttentionUnit(embed_dim)

        # --- 4. 核心支路 ---
        num_fields = 9 + 1 # 保持 10 个特征块
        self.cross_network = CrossNetworkV2(num_fields, embed_dim, num_layers=cross_layers)
        
        input_dim = num_fields * embed_dim
        self.mlp_layer1 = nn.Sequential(
            nn.Linear(input_dim, mlp_dims[0]),
            nn.BatchNorm1d(mlp_dims[0]),
            nn.PReLU(),
            nn.Dropout(0.2)
        )
        self.mlp_layer2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        
        # --- 5. PPNet 门控 ---
        self.ppnet_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.PReLU(),
            nn.Linear(32, mlp_dims[1]),
            nn.Sigmoid()
        )
        
        self.final_linear = nn.Linear(mlp_dims[1] + input_dim, 1)

    def forward(self, cat_features, seq_features, num_features):
        # A. 提取基础 Embedding
        u_emb = self.user_emb_layer(cat_features['user_idx'])
        i_emb = self.item_emb_layer(cat_features['item_idx'])
        
        # B. 执行 DIN 注意力池化 (针对历史物品序列)
        h_items = seq_features['hist_movie_ids']
        h_item_embs = self.item_emb_layer(h_items)
        mask_i = (h_items != 0).float()
        # 核心：根据当前 i_emb 动态激活历史序列
        din_pooled_emb = self.attention_unit(i_emb, h_item_embs, mask_i)
        
        # C. 执行题材静态 Mean Pooling
        h_genres = seq_features['hist_genre_ids']
        mask_g = (h_genres != 0).float().unsqueeze(-1)
        genre_pooled_emb = torch.sum(self.genre_emb_layer(h_genres) * mask_g, dim=1) / (torch.sum(mask_g, dim=1) + 1e-8)
        
        # D. 构造特征矩阵 [Batch, 10, 16]
        fields = [u_emb, i_emb, din_pooled_emb, genre_pooled_emb]
        for name in self.stat_feat_names:
            fields.append(self.stat_embs[name](cat_features[name]))
        
        num_v = torch.stack([num_features['genre_density'], num_features['genre_recent_match'], num_features['genre_rating_bias']], dim=-1)
        fields.append(F.relu(self.num_projection(num_v)))
        
        x0 = torch.stack(fields, dim=1)
        x_flat = x0.view(x0.size(0), -1)
        
        # 1. Cross 路
        cross_out = self.cross_network(x0).view(x0.size(0), -1)
        
        # 2. Deep 路 (带门控)
        d_out = self.mlp_layer1(x_flat)
        d_out = self.mlp_layer2(d_out)
        
        # PPNet 干预 (Sigmoid * 2)
        u_info = torch.cat([u_emb, self.stat_embs['user_rating_bucket'](cat_features['user_rating_bucket']), 
                            self.stat_embs['user_count_bucket'](cat_features['user_count_bucket'])], dim=-1).detach()
        gate_v = self.ppnet_gate(u_info) * 2
        d_out = F.relu(d_out * gate_v)
        
        # 3. 最终汇总
        combined = torch.cat([cross_out, d_out], dim=-1)
        return self.final_linear(combined).squeeze(-1)
