import pandas as pd
import numpy as np
import pickle
import os
import random
import re
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# --- 全局配置 ---
# 抽取 20% 的用户进行排序训练，以平衡训练速度与模型学习能力
USER_FRACTION = 0.2           
# 每位用户最近 100 个正样本 (Label=1)，确保排序模型学习的是用户最新的偏好
MAX_POS_PER_USER = 100        
# 每位用户最近 100 个真实差评 (Label=0)，提供最确定的负面偏好信号
MAX_REAL_NEG_PER_USER = 100   
# 特征计算时的快照截断长度：取最近 200 个点击以计算题材浓度，兼顾精度与计算性能
FEAT_SNAPSHOT_MAX = 200
# 输出数据存放目录
OUTPUT_DIR = 'data/ranking_processed/'

def calculate_genre_punch(target_item_genres, snapshot_history_items, item_idx2genres, snapshot_history_ratings):
    """
    核心特征工程：计算题材『组合拳』数值特征。
    包含：1. 兴趣浓度 (Density) | 2. 近期匹配 (Recent Match) | 3. 评分偏好 (Rating Bias)。
    """
    if not target_item_genres or not snapshot_history_items:
        return 0.0, 0.0, 0.5 # 默认中性值
    
    # 提取最近 200 个历史用于统计题材偏好分布
    hist_items_200 = snapshot_history_items[-FEAT_SNAPSHOT_MAX:]
    hist_ratings_200 = snapshot_history_ratings[-FEAT_SNAPSHOT_MAX:]
    
    # 初始化统计容器
    genre_counts = defaultdict(int)
    genre_total_score = defaultdict(float)
    genre_occur_count = defaultdict(int)
    
    for item, rating in zip(hist_items_200, hist_ratings_200):
        gs = item_idx2genres.get(item, [])
        for g in gs:
            genre_counts[g] += 1
            genre_total_score[g] += rating
            genre_occur_count[g] += 1
            
    total_genre_appearances = sum(genre_counts.values())
    
    # 1. 浓度特征 (Density)：当前候选电影题材在用户历史题材流中的占比
    density = sum(genre_counts[g] for g in target_item_genres) / (total_genre_appearances + 1e-8)
    
    # 2. 近期匹配 (Recent Match)：判断当前电影是否命中用户最近 5 个行为的题材（即时兴趣捕捉）
    recent_5_items = snapshot_history_items[-5:]
    recent_5_genres = set()
    for item in recent_5_items:
        recent_5_genres.update(item_idx2genres.get(item, []))
    recent_match = 1.0 if (set(target_item_genres) & recent_5_genres) else 0.0
    
    # 3. 评分偏好 (Rating Bias)：历史中该题材的平均得分，反映用户对该领域的『挑剔程度』
    relevant_scores = [genre_total_score[g] / (genre_occur_count[g] + 1e-8) for g in target_item_genres if g in genre_occur_count]
    rating_bias = np.mean(relevant_scores) / 5.0 if relevant_scores else 0.6 # 没看过给个中性及格分
    
    return density, recent_match, rating_bias

def process_ranking_samples(user_indices, pos_df, neg_ratings_all, movie_info, meta, year_hot_map):
    """
    并行处理单元：基于召回阶段的划分，为用户生产包含『组合拳数值特征』的精排样本。
    """
    results = []
    # 提取多进程共享的静态索引
    item_idx2genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    user_feat_map = meta['user_feat_map']
    item_idx2ts = movie_info['first_ts']    # 电影首映时间戳索引
    item_idx2year = movie_info['idx2year']  # 电影上映年份索引

    # 建立子进程局部的差评映射池
    user_neg_pool = neg_ratings_all[neg_ratings_all['user_idx'].isin(user_indices)].groupby('user_idx')
    
    for u_idx in user_indices:
        # A. 获取该用户划分好的正向点击样本 (Label=1)
        u_pos_samples = pos_df[pos_df['user_idx'] == u_idx]
        if len(u_pos_samples) == 0: continue
        
        # B. 获取该用户的真实负向记录 (Label=0)
        try:
            u_neg_real = user_neg_pool.get_group(u_idx).sort_values('timestamp').tail(MAX_REAL_NEG_PER_USER)
        except KeyError:
            u_neg_real = pd.DataFrame()
            
        # 获取用户高分序列及对应的评分 (用于快照特征计算)
        u_full_seq = meta['user_sequences'].get(u_idx, [])
        u_full_ratings = meta.get('user_ratings_seq', {}).get(u_idx, [4.0]*len(u_full_seq))
        
        # 用户全量交互集合，用于补位采样时的去重
        full_history_set = set(u_full_seq) | (set(u_neg_real['item_idx']) if not u_neg_real.empty else set())
        u_stats = user_feat_map.get(u_idx, {})

        def build_row(target_item, label, ts, pos_in_seq):
            """
            构造样本行：集成快照时序特征与题材组合拳。
            """
            i_feats = item_feat_map.get(target_item, {})
            # 1. 严格截取该行为发生时刻之前的 50 个高分点击作为历史 ID 特征
            hist_ids = (u_full_seq[:pos_in_seq][-50:] + [0]*50)[:50]
            # 2. 构造历史题材偏好特征
            hist_gs = []
            for h in hist_ids:
                if h != 0: hist_gs.extend(item_idx2genres.get(h, []))
            hist_gs_truncated = (hist_gs[-100:] + [0]*100)[:100]
            
            # 3. 实时计算题材组合拳数值特征 (浓度、近期匹配、偏好偏差)
            snap_items = u_full_seq[:pos_in_seq]
            snap_ratings = u_full_ratings[:pos_in_seq]
            density, recent_match, rating_bias = calculate_genre_punch(item_idx2genres.get(target_item, []), snap_items, item_idx2genres, snap_ratings)
            
            return {
                'user_idx': u_idx, 'item_idx': target_item, 'label': label,
                'year_bucket': i_feats.get('year_bucket', 0),
                'item_rating_bucket': i_feats.get('rating_bucket', 0),
                'item_count_bucket': i_feats.get('count_bucket', 0),
                'user_rating_bucket': u_stats.get('rating_bucket', 0), 
                'user_count_bucket': u_stats.get('count_bucket', 0),
                'hist_movie_ids': hist_ids, 'hist_genre_ids': hist_gs_truncated,
                # 题材组合拳数值特征
                'genre_density': density, 
                'genre_recent_match': recent_match, 
                'genre_rating_bias': rating_bias
            }

        # --- 第一步：处理每一个好评行为并执行补位采样 (1:2 比例补位) ---
        for _, row in u_pos_samples.iterrows():
            item_id, ts, pos_val = int(row['item_idx']), row.get('timestamp', 0), int(row['pos'])
            # 记录正样本行
            results.append(build_row(item_id, 1, ts, pos_val))
            
            # 执行补位负采样逻辑
            target_year = item_idx2year.get(item_id, 2000)
            # B1. 热门补位：从【当年+前一年】Top-20 热门榜中寻找未看过的电影
            hot_pool = year_hot_map.get(target_year, []) + year_hot_map.get(target_year-1, [])
            hot_neg = 0
            if hot_pool:
                random.shuffle(hot_pool)
                for cand in hot_pool:
                    # 严谨约束：用户没看过 且 负样本在该正向点击发生时已在系统中『存在』
                    if cand not in full_history_set and item_idx2ts.get(cand, 9e15) < ts:
                        hot_neg = cand; break
            if hot_neg: results.append(build_row(hot_neg, 0, ts, pos_val))
            
            # B2. 全局随机补位：提供广泛的基础负信号
            rand_neg = 0
            for _ in range(5):
                cand = random.randint(1, meta['item_count']-1)
                if cand not in full_history_set and item_idx2ts.get(cand, 9e15) < ts:
                    rand_neg = cand; break
            if rand_neg: results.append(build_row(rand_neg, 0, ts, pos_val))

        # --- 第二步：处理该用户的所有真实差评 (独立负样本行) ---
        for _, row in u_neg_real.iterrows():
            neg_item, neg_ts = int(row['item_idx']), row['timestamp']
            # 根据时间戳动态计算该差评时刻在正向点击序列中的切片位置
            pos_val = len([x for x in u_full_seq if item_idx2ts.get(x, 0) < neg_ts])
            results.append(build_row(neg_item, 0, neg_ts, pos_val))

    return results

def generate_ranking_dataset():
    """
    排序预处理主逻辑：读取召回阶段划分好的 train.csv，并行生产精排训练集。
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    random.seed(42); np.random.seed(42)

    print("正在加载召回元数据并建立时序索引...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    train_pos_df = pd.read_csv('data/processed/train.csv')
    # 读取原始评分数据用于差评挖掘和组合拳计算
    ratings = pd.read_csv('data/ml-32m/ratings.csv', usecols=['userId', 'movieId', 'rating', 'timestamp'], dtype={'rating': float, 'timestamp': int})
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # 映射 MovieID 与 UserID (需与召回模型完全对齐)
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    ratings['item_idx'] = ratings['movieId'].map(movie2id)
    unique_user_ids = ratings[ratings['rating'] >= 3.0]['userId'].unique()
    user2id = {uid: i for i, uid in enumerate(unique_user_ids)}
    ratings['user_idx'] = ratings['userId'].map(user2id)
    
    # 建立电影首映时间戳字典 (物理硬约束)
    movie_first_ts = ratings.groupby('item_idx')['timestamp'].min().to_dict()
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title); return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    item_idx2year = movies.set_index(movies['movieId'].map(movie2id))['year'].to_dict()
    
    # 预计算年度热门 Top-20 列表，用于 Hard Negative 补位采样
    item_counts = ratings.groupby('item_idx').size().to_dict()
    year_hot_map = {y: sorted(movies[movies['year']==y]['movieId'].map(movie2id).dropna().tolist(), 
                             key=lambda x: item_counts.get(x, 0), reverse=True)[:20] for y in movies['year'].unique() if y != 0}

    # 提取差评库
    neg_ratings_all = ratings[ratings['rating'] < 3.0].copy()
    # 采样指定比例的用户参与训练
    all_sampled_users = np.random.choice(list(user2id.values()), int(len(user2id) * USER_FRACTION), replace=False)
    
    movie_info = {'first_ts': movie_first_ts, 'idx2year': item_idx2year}
    num_workers = max(1, os.cpu_count() - 1)

    print(f"启动 {num_workers} 个并行 Worker 生产精排训练样本 (用户数: {len(all_sampled_users)})...")
    train_batches = np.array_split(all_sampled_users, num_workers)
    train_final = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_ranking_samples, b, train_pos_df, neg_ratings_all, movie_info, meta, year_hot_map) for b in train_batches]
        for f in tqdm(futures, desc="样本生产进度"): train_final.extend(f.result())
    
    # 保存至 Parquet
    print(f"训练样本总数: {len(train_final):,}")
    pd.DataFrame(train_final).to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'))
    print(f"排序训练数据保存成功！输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_ranking_dataset()
