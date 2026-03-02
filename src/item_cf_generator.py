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

def generate_item_cf_matrix(train_path='data/processed/train.csv', output_path='data/processed/item_cf_sim.pkl'):
    """
    基于训练集生成 ItemCF 相似度矩阵（带 IUF 惩罚与可选的位置权重）。
    """
    print(f"正在加载训练数据 (SIM_EXPONENT={SIM_EXPONENT}, PositionWeight={USE_POSITION_WEIGHT})...")
    df = pd.read_csv(train_path)
    
    # 1. 构建倒排索引：User -> List of Items (按时间顺序)
    print("构建用户行为倒排索引...")
    user_item_dict = df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    # 2. 计算共现矩阵与权重
    co_occurrence = defaultdict(lambda: defaultdict(float))
    item_count_dict = defaultdict(int)
    
    print("开始计算共现矩阵 (应用权重策略)...")
    for u_idx, items in tqdm(user_item_dict.items()):
        # 计算 IUF 权重
        u_weight = 1.0 / math.log1p(len(items))
        
        for idx_i, i in enumerate(items):
            item_count_dict[i] += 1
            for idx_j, j in enumerate(items):
                if i == j: continue
                
                # 最终贡献权重计算
                final_weight = u_weight
                if USE_POSITION_WEIGHT:
                    # 应用用户提供的公式：1 / ln(1 + |pos_i - pos_j|)
                    dist = abs(idx_i - idx_j)
                    # 避免对数 0 错误（由于 i!=j，dist 最小为 1）
                    final_weight *= (1.0 / math.log(1 + dist))
                
                co_occurrence[i][j] += final_weight
                
    # 3. 计算最终相似度并进行归一化
    print("计算相似度并执行 Top-100 截断...")
    item_sim_matrix = {}
    
    for i, related_items in tqdm(co_occurrence.items()):
        sim_list = []
        for j, weight in related_items.items():
            # 归一化公式：w_ij / (count_i * count_j)^SIM_EXPONENT
            sim = weight / math.pow(item_count_dict[i] * item_count_dict[j], SIM_EXPONENT)
            sim_list.append((j, sim))
        
        # 使用 (-score, item_id) 双重排序确保结果绝对稳定
        sim_list.sort(key=lambda x: (-x[1], x[0]))
        item_sim_matrix[i] = sim_list[:100]
        
    print(f"保存相似度字典至 {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(item_sim_matrix, f)
    print("ItemCF 相似度矩阵生成完成！")

if __name__ == "__main__":
    generate_item_cf_matrix()
