import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import math

def generate_item_cf_matrix(train_path='data/processed/train.csv', output_path='data/processed/item_cf_sim.pkl'):
    """
    基于训练集生成 ItemCF 相似度矩阵（带 IUF 惩罚）。
    """
    print("正在加载训练数据用于 ItemCF...")
    df = pd.read_csv(train_path)
    
    # 1. 构建倒排索引：User -> List of Items
    print("构建用户行为倒排索引...")
    user_item_dict = df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    # 2. 计算共现矩阵与 IUF 惩罚
    # co_occurrence[i][j] 存储物品 i 和 j 的共现得分
    co_occurrence = defaultdict(lambda: defaultdict(float))
    # item_count_dict 存储每个物品被多少人看过，用于分母归一化
    item_count_dict = defaultdict(int)
    
    print("开始计算共现矩阵 (应用 IUF 惩罚)...")
    for u_idx, items in tqdm(user_item_dict.items()):
        # 计算该用户的 IUF 权重
        # 看片量越大的用户，其贡献的权重越低
        u_weight = 1.0 / math.log1p(len(items))
        
        for i in items:
            item_count_dict[i] += 1
            for j in items:
                if i == j: continue
                co_occurrence[i][j] += u_weight
                
    # 3. 计算最终相似度并进行归一化
    print("计算余弦相似度并执行 Top-100 截断...")
    item_sim_matrix = {}
    
    for i, related_items in tqdm(co_occurrence.items()):
        # 针对每个物品 i，只保留最相似的前 100 个邻居，以节省内存
        # 归一化公式：w_ij / sqrt(count_i * count_j)
        sim_list = []
        for j, weight in related_items.items():
            sim = weight / math.sqrt(item_count_dict[i] * item_count_dict[j])
            sim_list.append((j, sim))
        
        # 降序排列，取前 100
        sim_list.sort(key=lambda x: x[1], reverse=True)
        item_sim_matrix[i] = sim_list[:100]
        
    print(f"保存相似度字典至 {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(item_sim_matrix, f)
    print("ItemCF 相似度矩阵生成完成！")

if __name__ == "__main__":
    generate_item_cf_matrix()
