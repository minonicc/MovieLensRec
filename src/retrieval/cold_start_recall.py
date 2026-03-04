import pandas as pd
import numpy as np
import pickle
import os
import re
from tqdm import tqdm
from collections import defaultdict

# --- 全局配置 ---
COLD_COUNT_LIMIT = 10  # 评分数少于 10 的定义为冷门电影
K_LIST = [100, 200, 500, 1000]

def get_year_score(movie_year, target_year):
    """
    计算年份贴合度得分。优先级：[target-3, target] > (target, target+3] > others
    """
    diff = movie_year - target_year
    if -3 <= diff <= 0:
        return 100 - abs(diff) 
    elif 0 < diff <= 3:
        return 80 - abs(diff)  
    else:
        return 50 - abs(diff)

def run_cold_start_recall():
    """
    执行基于题材画像的多级冷启动召回评估。
    采用定向采样逻辑，仅针对测试集中点击了冷门电影的用户进行性能测试。
    """
    print(f"正在加载元数据与全量电影信息 (冷门阈值: {COLD_COUNT_LIMIT})...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # --- 1. 题材字典动态提取 (核心修复) ---
    all_genres_set = set()
    for g_str in movies['genres'].str.split('|'): all_genres_set.update(g_str)
    all_genres_list = sorted(list(all_genres_set))
    genre2idx = {g: i for i, g in enumerate(all_genres_list)}
    print(f"检测到系统中共有 {len(all_genres_list)} 种题材。")

    # --- 2. 精准统计物品受关注度 ---
    item_real_counts = defaultdict(int)
    for seq in meta['user_sequences'].values():
        for item_idx in seq: item_real_counts[item_idx] += 1
            
    # --- 3. 电影属性预处理 (年份/索引) ---
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    movies['item_idx'] = movies['movieId'].map(movie2id)
    item_mean_dict = {idx: feat.get('rating_bucket', 0) for idx, feat in meta['item_feat_map'].items()}

    # --- 4. 准备冷门电影候选池 ---
    cold_movies = movies[movies['item_idx'].apply(lambda x: item_real_counts[x] < COLD_COUNT_LIMIT)].copy()
    cold_item_indices = cold_movies['item_idx'].values
    
    # 构造候选池题材矩阵
    cold_matrix = np.zeros((len(cold_movies), len(all_genres_list)))
    for i, (_, row) in enumerate(cold_movies.iterrows()):
        for g in row['genres'].split('|'):
            if g in genre2idx: cold_matrix[i, genre2idx[g]] = 1.0
    
    # L2 归一化电影向量
    norms = np.linalg.norm(cold_matrix, axis=1, keepdims=True)
    cold_matrix = np.divide(cold_matrix, norms, out=np.zeros_like(cold_matrix), where=norms!=0)
    print(f"冷门候选池大小: {len(cold_movies)} 部电影")

    # --- 5. 定向筛选评估目标 (Oracle 逻辑) ---
    # 找出测试集中真正点击了冷门电影的用户，只有他们才可能被本路径命中
    eval_df = test_df[test_df['item_idx'].apply(lambda x: item_real_counts[x] < COLD_COUNT_LIMIT)].copy()
    num_eval = len(eval_df)
    
    if num_eval == 0:
        print("错误：测试集中没有用户点击冷门电影，无法评估冷启动性能。")
        return

    print(f"定向评估启动：测试集中共有 {num_eval} 个冷启动目标用户。")

    # --- 6. 执行召回循环 ---
    user_sequences = meta['user_sequences']
    movie_genres = meta['movie_genres']
    hits = {k: 0 for k in K_LIST}

    for _, row in tqdm(eval_df.iterrows(), total=num_eval, desc="评估进度"):
        u_idx = int(row['user_idx'])
        target_item = int(row['item_idx'])
        pos = int(row['pos'])
        
        # A. 构造用户题材画像 (微平滑)
        history = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history)
        user_genre_counts = np.zeros(len(all_genres_list)) + 0.01 # 降低平滑噪音
        
        for j, h_item in enumerate(history):
            # 最近 10 次点击双倍权重
            weight = 2.0 if (len(history) - j) <= 10 else 1.0
            # 注意：此处从 meta['movie_genres'] 取的是原始编码，需转换为本脚本的 index
            g_ids = movie_genres.get(h_item, [])
            for gid in g_ids:
                # meta 中的 gid 对应 genre2id，需要映射到 GENRE_LIST 的位置
                # 这里我们直接用字符串匹配 row['genres'] 的逻辑构造矩阵，
                # 但在算用户侧时，最保险的是也从 movies 数据中取题材名。
                pass # 下面改用更直接的题材名累加逻辑
        
        # 重新采用基于题材名的稳健累加逻辑
        for j, h_item in enumerate(history):
            weight = 2.0 if (len(history) - j) <= 10 else 1.0
            # 找到这部电影的题材字符串
            m_genres = movies[movies['item_idx'] == h_item]['genres'].values
            if len(m_genres) > 0:
                for g_name in m_genres[0].split('|'):
                    if g_name in genre2idx: user_genre_counts[genre2idx[g_name]] += weight
        
        # L2 归一化
        u_norm = np.linalg.norm(user_genre_counts)
        user_vec = user_genre_counts / u_norm if u_norm > 0 else user_genre_counts

        # B. 题材匹配计算
        genre_scores = np.dot(cold_matrix, user_vec)

        # C. 多级排序与截断
        last_item_idx = history[-1] if history else 0
        last_year_info = movies[movies['item_idx'] == last_item_idx]['year'].values
        target_year = last_year_info[0] if len(last_year_info) > 0 else 2000

        rank_df = pd.DataFrame({
            'item_idx': cold_item_indices,
            'genre_score': genre_scores,
            'avg_rating': [item_mean_dict.get(idx, 0) for idx in cold_item_indices],
            'year': cold_movies['year'].values
        })
        
        # 排除已看过
        rank_df = rank_df[~rank_df['item_idx'].isin(history_set)]
        rank_df['year_score'] = rank_df['year'].apply(lambda y: get_year_score(y, target_year))
        
        # 排序：题材匹配 > 年份贴合 > 均分 (调整顺序，年份优先于均分)
        topk_all = rank_df.sort_values(['genre_score', 'year_score', 'avg_rating'], 
                                      ascending=[False, False, False])
        
        recall_ids = topk_all['item_idx'].values
        for k in K_LIST:
            if target_item in recall_ids[:k]: hits[k] += 1

    # --- 7. 输出报告 ---
    print("\n" + "="*50)
    print(f"冷启动召回定向评估报告 (针对真正的冷门点击者)")
    print(f"冷门定义: count < {COLD_COUNT_LIMIT} | 有效测试样本: {num_eval}")
    print("="*50)
    for k in K_LIST:
        print(f"Hit Rate @ {k:<4}: {hits[k] / num_eval:.2%}")
    print("="*50)
    print("注：该指标反映了『当用户确实想看冷门片时，该算法的捕捉概率』。")

if __name__ == "__main__":
    run_cold_start_recall()
