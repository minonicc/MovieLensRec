import pandas as pd
import numpy as np
import pickle
import os
import random
import re

# --- 全局配置 ---
# 保持用户指定的采样比例，以便进行一致的横向对比
USER_SAMPLE_FRACTION = 1.0 
# 困难负采样的随机性比例：30% 的样本将强制使用『同题材高热度未看』电影，增加负样本多样性
HARD_NEG_RANDOMNESS = 0.3

def preprocess(data_dir='data/ml-32m/', output_dir='data/processed/'):
    """
    数据预处理主函数：执行清洗、特征工程（年份/分桶）、ID映射、序列生成及数据集划分。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置随机种子以保证预处理采样的一致性
    random.seed(42)
    np.random.seed(42)

    print(f"正在加载数据 (采样比例: {USER_SAMPLE_FRACTION})...")
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    # 指定 dtype 以节省读取时的内存开销
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), 
                         dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int})

    # --- 1. 基础预处理 ---
    # 从标题中提取电影上映年份
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)

    # 映射 MovieID (0留给Padding，电影 ID 从 1 开始)
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    movies['item_idx'] = movies['movieId'].map(movie2id)
    
    # 计算全局流行度 (用于后续采样高热度负样本)
    movie_pop = ratings.groupby('movieId').size().to_dict()
    movies['popularity'] = movies['movieId'].map(movie_pop).fillna(0)

    # 处理题材
    all_genres = set()
    for g in movies['genres'].str.split('|'): all_genres.update(g)
    # 类型编码也从 1 开始，0 为 Padding
    genre2id = {g: i+1 for i, g in enumerate(sorted(list(all_genres)))}
    
    # 构建电影索引到类型列表的映射
    movie_genres = {movie2id[row['movieId']]: [genre2id[g] for g in row['genres'].split('|') if g in genre2id] 
                    for _, row in movies.iterrows()}

    # --- 2. 数据过滤与用户采样 ---
    # 分别提取正样本 (用于训练/评估) 和负向反馈 (用于困难负采样)
    # 注意：这里的正样本阈值已根据用户要求调整为 3.0
    pos_ratings_all = ratings[ratings['rating'] >= 3.0].copy()
    neg_feedback_all = ratings[ratings['rating'] < 3.0].copy()

    if USER_SAMPLE_FRACTION < 1.0:
        unique_users = pos_ratings_all['userId'].unique()
        sampled_users = np.random.choice(unique_users, int(len(unique_users) * USER_SAMPLE_FRACTION), replace=False)
        pos_ratings = pos_ratings_all[pos_ratings_all['userId'].isin(sampled_users)].copy()
        neg_feedback = neg_feedback_all[neg_feedback_all['userId'].isin(sampled_users)].copy()
    else:
        pos_ratings = pos_ratings_all
        neg_feedback = neg_feedback_all

    # 映射 UserID
    user2id = {id: i for i, id in enumerate(pos_ratings['userId'].unique())}
    pos_ratings['user_idx'] = pos_ratings['userId'].map(user2id)
    pos_ratings['item_idx'] = pos_ratings['movieId'].map(movie2id)
    neg_feedback['user_idx'] = neg_feedback['userId'].map(user2id)
    neg_feedback['item_idx'] = neg_feedback['movieId'].map(movie2id)
    
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
    # 年份分桶 (针对 1990-2023 优化，共 10 个有效区间)
    movies['year_bucket'] = pd.cut(movies['year'], bins=[0, 1990, 1995, 2000, 2005, 2010, 2013, 2016, 2019, 2022, 2030], labels=False)
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

    # --- 5. 构造采样池：差评池与题材热度池 ---
    print("构建采样池...")
    # 用户真实差评库 (<3.0) 集合，用于 O(1) 查找
    user_dislikes = neg_feedback.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    # 题材热度池：每个题材下，按流行度从高到低存储电影 ID
    genre_to_hot_items = {}
    for g_id in genre2id.values():
        # 找出该题材下的所有电影，并按 popularity 排序
        # 通过索引查找题材名称进行过滤
        genre_name = [name for name, i in genre2id.items() if i == g_id][0]
        items_in_g = movies[movies['genres'].str.contains(re.escape(genre_name))]
        sorted_items = items_in_g.sort_values('popularity', ascending=False)['item_idx'].tolist()
        genre_to_hot_items[g_id] = sorted_items[:200] # 只保留最热门的前 200 个，增加负采样难度

    # --- 6. 构造样本并执行采样 (90最近+10随机) ---
    print("执行样本采样逻辑 (引入 30% 随机性与热度兜底)...")
    # 聚合每个用户的物品点击序列
    user_sequences = pos_ratings.groupby('user_idx')['item_idx'].apply(list).to_dict()
    # 额外存储评分序列，用于将评分同步记录到数据集
    user_ratings_seq = pos_ratings.groupby('user_idx')['rating'].apply(list).to_dict()
    
    train_samples, val_samples, test_samples = [], [], []
    for u_idx, seq in user_sequences.items():
        if len(seq) < 3: continue
        ratings_seq = user_ratings_seq[u_idx]
        dislikes = user_dislikes.get(u_idx, set())
        # 用户接触过的所有物品集合，用于采样时的去重
        full_history_set = set(seq) | dislikes 
        
        # 留一法：最后1个测试，倒数第2个验证 (同步保存原始 rating 用于后续分析)
        test_samples.append([u_idx, seq[-1], len(seq)-1, ratings_seq[-1]])
        val_samples.append([u_idx, seq[-2], len(seq)-2, ratings_seq[-2]])
        
        # 训练集池子 (排除掉最后两个目标)
        pool_indices = list(range(1, len(seq)-2))
        
        # 采样逻辑：每个用户最多 100 个正样本 (90最近 + 10随机)
        if len(pool_indices) <= 100:
            selected = pool_indices
        else:
            selected = sorted(pool_indices[-90:] + random.sample(pool_indices[:-90], 10))
            
        for i in selected:
            pos_item = seq[i]
            pos_item_genres = movie_genres.get(pos_item, [])
            hard_neg = 0
            
            # --- 核心改进：引入 30% 随机性与高热度负采样 ---
            # 70% 概率且有真实差评时，走真实差评逻辑
            if dislikes and random.random() > HARD_NEG_RANDOMNESS:
                # 尝试在该用户的真实差评中找同题材的
                for d_item in dislikes:
                    if any(g in movie_genres.get(d_item, []) for g in pos_item_genres):
                        hard_neg = d_item; break
                # 兜底：随机选一个该用户的真实差评
                if hard_neg == 0: hard_neg = random.choice(list(dislikes))
            
            # 30% 概率，或用户无差评记录时，选一个同题材的高流行度电影
            if hard_neg == 0 and pos_item_genres:
                # 优先使用第一个题材的主热度池
                hot_candidates = genre_to_hot_items.get(pos_item_genres[0], [])
                if hot_candidates:
                    # 从该题材最火的电影中，选一个用户没看过的作为困难负样本
                    for c in hot_candidates:
                        if c not in full_history_set:
                            hard_neg = c; break
            
            # 极致兜底：如果以上逻辑都未命中，随机选一个电影
            if hard_neg == 0:
                hard_neg = random.randint(1, len(movie2id))

            train_samples.append([u_idx, pos_item, i, ratings_seq[i], hard_neg])

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
    
    # 保存结果，确保包含 rating 和 hard_neg_idx
    pd.DataFrame(train_samples, columns=['user_idx', 'item_idx', 'pos', 'rating', 'hard_neg_idx']).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame(val_samples, columns=['user_idx', 'item_idx', 'pos', 'rating']).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    pd.DataFrame(test_samples, columns=['user_idx', 'item_idx', 'pos', 'rating']).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"预处理完成！总训练样本数: {len(train_samples)}")

if __name__ == "__main__":
    preprocess()
