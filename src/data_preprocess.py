import pandas as pd
import numpy as np
import pickle
import os
import random
import re

# --- 全局配置 ---
# 保持用户指定的采样比例，以便进行一致的横向对比
USER_SAMPLE_FRACTION = 0.2

def preprocess(data_dir='data/ml-32m/', output_dir='data/processed/'):
    # 设置随机种子以保证预处理采样的一致性
    random.seed(42)
    np.random.seed(42)
    """
    数据预处理主函数：执行清洗、特征工程（年份/分桶）、ID映射、序列生成及数据集划分。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在加载数据 (采样比例: {USER_SAMPLE_FRACTION})...")
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), 
                         dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int})

    # --- 1. 基础预处理 ---
    # 从标题中提取电影上映年份
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)

    # 映射 MovieID (0留给Padding)
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    movies['item_idx'] = movies['movieId'].map(movie2id)
    
    # 处理题材
    all_genres = set()
    for g in movies['genres'].str.split('|'): all_genres.update(g)
    genre2id = {g: i+1 for i, g in enumerate(sorted(list(all_genres)))}
    movie_genres = {movie2id[row['movieId']]: [genre2id[g] for g in row['genres'].split('|') if g in genre2id] 
                    for _, row in movies.iterrows()}

    # --- 2. 数据过滤与用户采样 ---
    pos_ratings = ratings[ratings['rating'] >= 3.0].copy()
    if USER_SAMPLE_FRACTION < 1.0:
        unique_users = pos_ratings['userId'].unique()
        sampled_users = np.random.choice(unique_users, int(len(unique_users) * USER_SAMPLE_FRACTION), replace=False)
        pos_ratings = pos_ratings[pos_ratings['userId'].isin(sampled_users)].copy()

    # 映射 UserID
    user2id = {id: i for i, id in enumerate(pos_ratings['userId'].unique())}
    pos_ratings['user_idx'] = pos_ratings['userId'].map(user2id)
    pos_ratings['item_idx'] = pos_ratings['movieId'].map(movie2id)
    
    # 严格按时间排序，这是计算快照和消除穿越的基础
    print("生成用户序列...")
    pos_ratings.sort_values(['user_idx', 'timestamp'], ascending=[True, True], inplace=True)
    
    # --- 3. 计算训练集快照统计特征 (消除穿越) ---
    print("计算快照统计特征...")
    # 标注出每个用户序列中的相对位置。由于已排序，cumcount(ascending=False) 0代表最后一条，1代表倒数第二条
    pos_ratings['item_rank_inv'] = pos_ratings.groupby('user_idx').cumcount(ascending=False)
    
    # 只有排名 >= 2 的记录属于训练集（即排除了最后两条验证/测试记录）
    train_ratings_mask = pos_ratings['item_rank_inv'] >= 2
    train_ratings_only = pos_ratings[train_ratings_mask]

    # 基于训练记录计算电影统计量：均分与受关注次数
    item_stats = train_ratings_only.groupby('item_idx')['rating'].agg(['mean', 'count']).reset_index()
    # 基于训练记录计算用户统计量：均分与评价总次数
    user_stats = train_ratings_only.groupby('user_idx')['rating'].agg(['mean', 'count']).reset_index()

    # --- 4. 分桶 (Bucketing) 逻辑 ---
    print("执行特征分桶...")
    # 优化点：使用 labels=False 返回整数索引，避开 Categorical fillna 报错
    # 年份分桶 (10个桶)
    movies['year_bucket'] = pd.cut(movies['year'], bins=[0, 1990, 1995, 2000, 2005, 2010, 2013, 2016, 2019, 2022, 2030], 
                                   labels=False)
    movies['year_bucket'] = (movies['year_bucket'] + 1).fillna(0).astype(int)
    
    # 均分分桶 (0-5分，0.5一个桶，共10个)
    item_stats['rating_bucket'] = pd.cut(item_stats['mean'], bins=np.linspace(0, 5, 11), labels=False)
    item_stats['rating_bucket'] = (item_stats['rating_bucket'] + 1).fillna(0).astype(int)
    
    user_stats['rating_bucket'] = pd.cut(user_stats['mean'], bins=np.linspace(0, 5, 11), labels=False)
    user_stats['rating_bucket'] = (user_stats['rating_bucket'] + 1).fillna(0).astype(int)

    # 次数分桶 (Log Scale 处理长尾，分10个桶)
    item_stats['count_bucket'] = pd.cut(np.log1p(item_stats['count']), bins=10, labels=False)
    item_stats['count_bucket'] = (item_stats['count_bucket'] + 1).fillna(0).astype(int)
    
    user_stats['count_bucket'] = pd.cut(np.log1p(user_stats['count']), bins=10, labels=False)
    user_stats['count_bucket'] = (user_stats['count_bucket'] + 1).fillna(0).astype(int)

    # 整理为查表字典
    item_feat_map = movies.set_index('item_idx')[['year_bucket']].to_dict('index')
    for _, row in item_stats.iterrows():
        idx = int(row['item_idx'])
        if idx in item_feat_map:
            item_feat_map[idx].update({'rating_bucket': int(row['rating_bucket']), 'count_bucket': int(row['count_bucket'])})
    
    user_feat_map = user_stats.set_index('user_idx')[['rating_bucket', 'count_bucket']].to_dict('index')

    # --- 5. 构造样本并采样 (90最近+10随机) ---
    print("执行样本采样逻辑...")
    user_sequences = pos_ratings.groupby('user_idx')['item_idx'].apply(list).to_dict()
    # 额外存储每个样本对应的原始评分，用于后续质量分析
    user_ratings_seq = pos_ratings.groupby('user_idx')['rating'].apply(list).to_dict()
    
    train_samples, val_samples, test_samples = [], [], []
    for u_idx, seq in user_sequences.items():
        if len(seq) < 3: continue
        ratings_seq = user_ratings_seq[u_idx]
        
        # 留一法：最后1个测试，倒数第2个验证 (加入原始评分)
        test_samples.append([u_idx, seq[-1], len(seq)-1, ratings_seq[-1]])
        val_samples.append([u_idx, seq[-2], len(seq)-2, ratings_seq[-2]])
        
        # 训练集池子
        pool = list(range(1, len(seq)-2))
        if len(pool) <= 100:
            selected = pool
        else:
            selected = sorted(pool[-90:] + random.sample(pool[:-90], 10))
            
        for i in selected:
            train_samples.append([u_idx, seq[i], i, ratings_seq[i]])

    print("保存预处理结果...")
    meta = {
        'user_count': len(user2id),
        'item_count': len(movie2id) + 1,
        'genre_count': len(genre2id) + 1,
        'movie_genres': movie_genres,
        'user_sequences': user_sequences,
        'item_feat_map': item_feat_map,
        'user_feat_map': user_feat_map
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    # 保存时包含 rating 列
    pd.DataFrame(train_samples, columns=['user_idx', 'item_idx', 'pos', 'rating']).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame(val_samples, columns=['user_idx', 'item_idx', 'pos', 'rating']).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    pd.DataFrame(test_samples, columns=['user_idx', 'item_idx', 'pos', 'rating']).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"预处理完成！总训练正样本数: {len(train_samples)}")

if __name__ == "__main__":
    preprocess()
