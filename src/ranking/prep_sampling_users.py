import pandas as pd
import numpy as np
import pickle
import os
import random

def prepare_ranking_users():
    """
    筛选 4 万名排序训练用户：包含 25% 活跃长序列用户 + 75% 随机用户。
    同步提取他们的评分序列，以支持题材评分偏好特征。
    """
    print("正在加载元数据...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    user_sequences = meta['user_sequences']
    # 尝试加载评分序列，如果没有，则去原始 ratings.csv 中提取
    user_ratings_seq = meta.get('user_ratings_seq', {})
    
    if not user_ratings_seq:
        print("Meta 中未发现评分序列，正在从原始 ratings.csv 提取...")
        ratings = pd.read_csv('data/ml-32m/ratings.csv', usecols=['userId', 'rating', 'timestamp'], dtype={'rating': float})
        # 建立与召回阶段一致的 user_idx 映射
        unique_users = ratings[ratings['rating'] >= 3.0]['userId'].unique()
        uid2idx = {uid: i for i, uid in enumerate(unique_users)}
        # 仅保留正向反馈的评分序列用于快照计算
        pos_ratings = ratings[ratings['rating'] >= 3.0].copy()
        pos_ratings['user_idx'] = pos_ratings['userId'].map(uid2idx)
        pos_ratings = pos_ratings.sort_values(['user_idx', 'timestamp'])
        user_ratings_seq = pos_ratings.groupby('user_idx')['rating'].apply(list).to_dict()

    # 过滤掉交互记录过少的用户 (少于 5 条的对排序训练无意义)
    all_users = [u for u in user_sequences.keys() if len(user_sequences[u]) >= 5]
    
    # 1. 识别长序列用户 (> 150 次点击)
    long_seq_users = [u for u in all_users if len(user_sequences[u]) > 150]
    other_users = [u for u in all_users if len(user_sequences[u]) <= 150]
    
    print(f"检测到活跃用户数: {len(long_seq_users)}, 普通用户数: {len(other_users)}")
    
    # 2. 抽样 40,000 人
    # 25% 活跃 = 10,000 人
    # 75% 普通 = 30,000 人
    random.seed(42)
    sampled_long = random.sample(long_seq_users, min(10000, len(long_seq_users)))
    sampled_others = random.sample(other_users, 40000 - len(sampled_long))
    
    ranking_users = sampled_long + sampled_others
    random.shuffle(ranking_users)
    
    # 3. 封装数据
    user_pack = {
        'user_ids': ranking_users,
        'user_sequences': {u: user_sequences[u] for u in ranking_users},
        'user_ratings_seq': {u: user_ratings_seq.get(u, [4.0]*len(user_sequences[u])) for u in ranking_users}
    }
    
    output_path = 'data/ranking_processed/ranking_user_pack.pkl'
    os.makedirs('data/ranking_processed', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(user_pack, f)
        
    print(f"黄金用户包生成完成！共 {len(ranking_users)} 人。")
    print(f"活跃用户占比: {len(sampled_long)/40000:.1%}")
    print(f"数据保存在: {output_path}")

if __name__ == "__main__":
    prepare_ranking_users()
