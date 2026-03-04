import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import math

# --- 全局配置 ---
# 相似度分母的幂次：用于控制对热门物品的惩罚力度。
# 0.5 代表标准余弦相似度；当前实验设置为 0.6。
SIM_EXPONENT = 0.6

# 是否启用位置权重 (Time-aware Weighting)：
# 启用后将根据物品在用户序列中的相对距离进行衰减，强化短期连续兴趣。
# 权重公式：1 / ln(1 + distance)
USE_POSITION_WEIGHT = True

# 是否启用评分权重 (Rating-consistency Weighting)：
# 启用后将根据两部电影的评分差异进行惩罚，评分越接近（偏好程度越一致），权重越高。
# 权重公式：1.0 - ALPHA * |rating_i - rating_j|
USE_RATING_WEIGHT = True
RATING_ALPHA = 0.25

def generate_item_cf_matrix(train_path='data/processed/train.csv', output_path='data/processed/item_cf_sim.pkl'):
    """
    基于训练集生成 ItemCF 相似度矩阵。
    集成了 IUF 惩罚、位置权重、评分一致性权重及热门打压幂次。
    """
    print(f"正在加载训练数据 (SIM_EXPONENT={SIM_EXPONENT}, PosWeight={USE_POSITION_WEIGHT}, RatingWeight={USE_RATING_WEIGHT})...")
    
    # 加载 item_idx 和 rating 列
    df = pd.read_csv(train_path, usecols=['user_idx', 'item_idx', 'rating'])
    
    # 1. 构建倒排索引：User -> List of (Item, Rating)
    print("构建用户行为倒排索引 (包含评分)...")
    # 先按 user 分组，将 item_idx 和 rating 聚合成元组列表
    user_data_dict = df.groupby('user_idx').apply(lambda x: list(zip(x['item_idx'], x['rating']))).to_dict()
    
    # 2. 计算共现矩阵与多重权重
    co_occurrence = defaultdict(lambda: defaultdict(float))
    item_count_dict = defaultdict(int)
    
    print("开始计算共现矩阵 (执行三重加权逻辑)...")
    for u_idx, data_list in tqdm(user_data_dict.items()):
        # 计算该用户的 IUF 权重：看片量越大的用户，其贡献的基础相似度越低
        u_weight = 1.0 / math.log1p(len(data_list))
        
        # 遍历用户历史中的每一个物品
        for idx_i, (item_i, rating_i) in enumerate(data_list):
            item_count_dict[item_i] += 1
            
            # 与序列中的其他物品计算共现贡献
            for idx_j, (item_j, rating_j) in enumerate(data_list):
                if item_i == item_j: continue
                
                # 初始化最终贡献权重
                final_weight = u_weight
                
                # A. 应用位置权重 (Time-aware)
                if USE_POSITION_WEIGHT:
                    dist = abs(idx_i - idx_j)
                    final_weight *= (1.0 / math.log(1 + dist))
                
                # B. 应用评分权重 (Consistency)
                if USE_RATING_WEIGHT:
                    rating_diff = abs(rating_i - rating_j)
                    # 偏好一致性权重计算：1.0 - 0.1 * 评分差
                    final_weight *= (1.0 - RATING_ALPHA * rating_diff)
                
                co_occurrence[item_i][item_j] += final_weight
                
    # 3. 计算最终相似度并进行归一化
    print("计算相似度并执行 Top-100 截断...")
    item_sim_matrix = {}
    
    for i, related_items in tqdm(co_occurrence.items()):
        sim_list = []
        for j, weight in related_items.items():
            # 归一化公式：w_ij / (count_i * count_j)^SIM_EXPONENT
            # SIM_EXPONENT 控制对热门物品的惩罚力度
            sim = weight / math.pow(item_count_dict[i] * item_count_dict[j], SIM_EXPONENT)
            sim_list.append((j, sim))
        
        # 使用 (-score, item_id) 双重排序确保结果唯一确定，增强实验可复现性
        sim_list.sort(key=lambda x: (-x[1], x[0]))
        item_sim_matrix[i] = sim_list[:100]
        
    print(f"保存相似度字典至 {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(item_sim_matrix, f)
    print("ItemCF 相似度矩阵生成完成！")

if __name__ == "__main__":
    generate_item_cf_matrix()
