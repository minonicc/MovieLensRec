import pandas as pd
import numpy as np
import pickle
import os
import random

# --- 全局配置 ---
USER_SAMPLE_FRACTION = 0.2  # 调试比例：0.1，全量训练请设为 1.0

def preprocess(data_dir='data/ml-32m/', output_dir='data/processed/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在加载数据 (正样本阈值: 3.0, 采样比例: {USER_SAMPLE_FRACTION})...")
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), 
                         dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int})

    # 1. 筛选正样本：按照您的要求，改为 3.0 分及以上
    pos_ratings = ratings[ratings['rating'] >= 3.0].copy()
    
    # 2. 按用户维度采样
    if USER_SAMPLE_FRACTION < 1.0:
        unique_users = pos_ratings['userId'].unique()
        sampled_users = np.random.choice(unique_users, int(len(unique_users) * USER_SAMPLE_FRACTION), replace=False)
        pos_ratings = pos_ratings[pos_ratings['userId'].isin(sampled_users)].copy()

    print("映射 ID...")
    user2id = {id: i for i, id in enumerate(pos_ratings['userId'].unique())}
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())} # 0为Padding
    
    pos_ratings['user_idx'] = pos_ratings['userId'].map(user2id)
    pos_ratings['item_idx'] = pos_ratings['movieId'].map(movie2id)
    
    # 处理题材
    all_genres = set()
    for g in movies['genres'].str.split('|'): all_genres.update(g)
    genre2id = {g: i+1 for i, g in enumerate(sorted(list(all_genres)))}
    movie_genres = {movie2id[row['movieId']]: [genre2id[g] for g in row['genres'].split('|') if g in genre2id] 
                    for _, row in movies.iterrows()}

    print("执行采样逻辑：每个用户最多 100 个训练样本 (90最近 + 10随机)...")
    pos_ratings.sort_values(['user_idx', 'timestamp'], ascending=[True, True], inplace=True)
    user_sequences = pos_ratings.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    train_samples, val_samples, test_samples = [], [], []

    for u_idx, seq in user_sequences.items():
        if len(seq) < 3: continue
        
        # 留一法：最后1个测试，倒数第2个验证
        test_samples.append([u_idx, seq[-1], len(seq)-1])
        val_samples.append([u_idx, seq[-2], len(seq)-2])
        
        # 训练集池子 (排除掉最后两个目标)
        train_pool_indices = list(range(1, len(seq)-2))
        
        if len(train_pool_indices) <= 100:
            selected_indices = train_pool_indices
        else:
            recent_indices = train_pool_indices[-90:]
            remaining_indices = train_pool_indices[:-90]
            random_indices = random.sample(remaining_indices, 10)
            selected_indices = sorted(recent_indices + random_indices)
            
        for i in selected_indices:
            train_samples.append([u_idx, seq[i], i])

    print("保存元数据...")
    meta = {
        'user_count': len(user2id),
        'item_count': len(movie2id) + 1,
        'genre_count': len(genre2id) + 1,
        'movie_genres': movie_genres,
        'user_sequences': user_sequences 
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    pd.DataFrame(train_samples, columns=['user_idx', 'item_idx', 'pos']).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame(val_samples, columns=['user_idx', 'item_idx', 'pos']).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    pd.DataFrame(test_samples, columns=['user_idx', 'item_idx', 'pos']).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"预处理完成！正样本数(评分>=3.0): {len(train_samples)}")

if __name__ == "__main__":
    preprocess()
