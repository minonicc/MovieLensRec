import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    """
    精排模型核心类：DeepFM (Factorization Machine + Deep Neural Network)。
    
    设计理念：
    1. 参数共享 (Parameter Sharing)：候选电影 ID 与用户历史序列 ID 共享物理 Embedding 矩阵，实现语义空间的绝对对齐。
    2. 低阶建模 (FM)：显式学习特征间的二阶内积交互，捕捉如『题材 x 年份』的组合规律。
    3. 高阶建模 (Deep)：利用多层 MLP 学习特征间复杂的非线性交互。
    4. 特征对齐：集成了离散 ID、分桶统计量、动态行为序列以及题材『组合拳』数值特征。
    """
    def __init__(self, feature_dims, embed_dim=16, mlp_dims=[128, 64]):
        """
        :param feature_dims: 字典 {特征名: 词表大小}，包含 user, item, stats 及序列特征的空间维度
        :param embed_dim: 每个离散特征映射后的向量维度 (K)
        :param mlp_dims: 高阶 Deep 部分各层神经元数
        """
        super(DeepFM, self).__init__()
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        
        # --- 1. 核心 Embedding 共享矩阵定义 ---
        # [共享] 电影 ID 空间：item_idx 与 hist_movie_ids 共享此表
        self.item_emb_layer = nn.Embedding(feature_dims['item_idx'], embed_dim, padding_idx=0)
        # [共享] 题材 ID 空间：用于表征用户的历史题材偏好分布
        self.genre_emb_layer = nn.Embedding(feature_dims['hist_genre_ids'], embed_dim, padding_idx=0)
        # [独立] 用户 ID 空间
        self.user_emb_layer = nn.Embedding(feature_dims['user_idx'], embed_dim)
        
        # [分桶] 统计类特征：年份、均分桶、次数桶使用独立的特征映射
        self.stat_feat_names = [
            'year_bucket', 'item_rating_bucket', 'item_count_bucket', 
            'user_rating_bucket', 'user_count_bucket'
        ]
        self.stat_embs = nn.ModuleDict({
            name: nn.Embedding(feature_dims[name], embed_dim) for name in self.stat_feat_names
        })

        # --- 2. 一阶线性权重层 (同样执行参数共享，模拟 LR 贡献) ---
        self.item_linear = nn.Embedding(feature_dims['item_idx'], 1, padding_idx=0)
        self.genre_linear = nn.Embedding(feature_dims['hist_genre_ids'], 1, padding_idx=0)
        self.user_linear = nn.Embedding(feature_dims['user_idx'], 1)
        self.stat_linears = nn.ModuleDict({
            name: nn.Embedding(feature_dims[name], 1) for name in self.stat_feat_names
        })

        # --- 3. Deep MLP 高阶层 ---
        # 输入维度计算：
        # 离散/序列块：User(1) + Item(1) + Stats(5) + Hist_Item(1) + Hist_Genre(1) = 9 个块
        # 数值块：题材组合拳 (3 个数值特征)
        input_dim = 9 * embed_dim + 3
        
        layers = []
        curr_dim = input_dim
        for next_dim in mlp_dims:
            layers.append(nn.Linear(curr_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim)) # 加入批归一化，解决深层网络梯度弥散
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) # 引入 Dropout 抑制 32M 数据在大规模参数下的过拟合
            curr_dim = next_dim
        self.mlp = nn.Sequential(*layers)
        # 最终输出层，将所有信号映射至 1 维 Logit 空间
        self.mlp_out = nn.Linear(curr_dim, 1)

    def _get_embeddings(self, cat_features, seq_features):
        """
        特征路由辅助函数：实现各特征到 Embedding 空间的精确映射与均值池化逻辑。
        """
        linears, embs = [], []
        
        # 1. 映射 User 特征
        u_idx = cat_features['user_idx']
        linears.append(self.user_linear(u_idx))
        embs.append(self.user_emb_layer(u_idx))
        
        # 2. 映射候选 Item 特征
        i_idx = cat_features['item_idx']
        linears.append(self.item_linear(i_idx))
        embs.append(self.item_emb_layer(i_idx))
        
        # 3. 循环处理各分桶统计特征
        for name in self.stat_feat_names:
            idx = cat_features[name]
            linears.append(self.stat_linears[name](idx))
            embs.append(self.stat_embs[name](idx))
            
        # 4. 映射历史物品序列 (物理共享 Item 权重)
        h_items = seq_features['hist_movie_ids']
        mask_i = (h_items != 0).float().unsqueeze(-1)
        # 线性池化
        h_i_lin = self.item_linear(h_items)
        linears.append(torch.sum(h_i_lin * mask_i, dim=1) / (torch.sum(mask_i, dim=1) + 1e-8))
        # 二阶池化
        h_i_emb = self.item_emb_layer(h_items)
        embs.append(torch.sum(h_i_emb * mask_i, dim=1) / (torch.sum(mask_i, dim=1) + 1e-8))
        
        # 5. 映射历史题材序列 (物理共享题材权重)
        h_genres = seq_features['hist_genre_ids']
        mask_g = (h_genres != 0).float().unsqueeze(-1)
        # 线性池化
        h_g_lin = self.genre_linear(h_genres)
        linears.append(torch.sum(h_g_lin * mask_g, dim=1) / (torch.sum(mask_g, dim=1) + 1e-8))
        # 二阶池化
        h_g_emb = self.genre_emb_layer(h_genres)
        embs.append(torch.sum(h_g_emb * mask_g, dim=1) / (torch.sum(mask_g, dim=1) + 1e-8))
        
        return linears, embs

    def forward(self, cat_features, seq_features, num_features):
        """
        前向传播主流程。
        """
        # A. 提取所有输入特征的 Linear 与 Embedding 分量
        linear_list, emb_list = self._get_embeddings(cat_features, seq_features)
        
        # --- 步骤 1: 计算一阶线性部分 (Linear Logit) ---
        # 将所有一阶贡献拼接并求和
        linear_logit = torch.sum(torch.cat(linear_list, dim=1), dim=1, keepdim=True)

        # --- 步骤 2: 计算二阶 FM 部分 (FM Logit) ---
        # 构造特征池张量 [Batch, Num_Fields, Embed_Dim]
        fm_input = torch.stack(emb_list, dim=1)
        
        # 采用 FM 经典优化公式实现二阶交互：0.5 * sum( (sum V_i)^2 - sum(V_i^2) )
        # 该公式的核心在于将全排列交叉转变为基于和的平方运算，大幅提升训练速度
        sum_of_square = torch.pow(torch.sum(fm_input, dim=1), 2)
        square_of_sum = torch.sum(torch.pow(fm_input, 2), dim=1)
        fm_logit = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1, keepdim=True)

        # --- 步骤 3: 计算高阶 Deep 部分 (Deep Logit) ---
        # 将 9 个特征块的 Embedding 向量展开为一维向量
        deep_input = fm_input.view(fm_input.size(0), -1)
        # 注入题材『组合拳』数值特征：浓度、匹配度、偏好评分
        num_feat_tensor = torch.stack([
            num_features['genre_density'],
            num_features['genre_recent_match'],
            num_features['genre_rating_bias']
        ], dim=-1)
        # 强特征拼接进入 MLP
        deep_input = torch.cat([deep_input, num_feat_tensor], dim=-1)
        deep_logit = self.mlp_out(self.mlp(deep_input))

        # --- 步骤 4: 汇总输出 ---
        # 返回原始 Logits 供损失函数计算。模型训练完毕后，需通过 Sigmoid 转化为预估点击率
        return (linear_logit + fm_logit + deep_logit).squeeze(-1)
