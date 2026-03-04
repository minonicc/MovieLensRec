import pandas as pd
import numpy as np
import pickle
import os
import re
from tqdm import tqdm
from collections import defaultdict

def run_year_popularity_recall(k=50):
    """
    专门的『年份热门召回』独立评估脚本：
    针对用户最后一次观影的时代背景，召回前后 5 年内的 Top-K 热门电影。
    """
    print(f"正在启动年份热门召回评估 (K={k})...")
    
    # 1. 加载元数据
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # 提取年份
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    
    # 映射与索引
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    movies['item_idx'] = movies['movieId'].map(movie2id)
    id2year = movies.set_index('item_idx')['year'].to_dict()

    # 2. 统计电影流行度 (全量训练集)
    item_counts = defaultdict(int)
    for seq in meta['user_sequences'].values():
        for item_idx in seq: item_counts[item_idx] += 1
    
    # 3. 预处理年度热门榜单
    print("生成年度热门索引...")
    year_hot_map = {}
    for year in movies['year'].unique():
        if year == 0: continue
        # 取该年份最火的前 100 部备用
        year_hot_map[year] = movies[movies['year'] == year].assign(pop=movies['item_idx'].map(item_counts)).sort_values('pop', ascending=False)['item_idx'].tolist()[:100]

    # 4. 执行评估
    user_sequences = meta['user_sequences']
    total_users = len(test_df)
    hit_count = 0
    
    print(f"正在对 {total_users} 个用户执行召回分析...")
    for _, row in tqdm(test_df.iterrows(), total=total_users):
        u_idx = int(row['user_idx'])
        target = int(row['item_idx'])
        pos = int(row['pos'])
        
        # 历史过滤
        history = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history)
        
        # 参考年份
        ref_year = id2year.get(history[-1], 2000) if history else 2000
        
        # 构造窗口候选集 [Y-2, Y+2]
        window_candidates = []
        for y in range(ref_year - 2, ref_year + 3):
            window_candidates.extend(year_hot_map.get(y, []))
        
        # 按全局流行度重新排序
        window_candidates = sorted(list(set(window_candidates)), 
                                  key=lambda x: item_counts.get(x, 0), 
                                  reverse=True)
        
        # 截取 Top-K
        recall_list = []
        for m_idx in window_candidates:
            if m_idx not in history_set:
                recall_list.append(m_idx)
                if len(recall_list) >= k: break
        
        if target in set(recall_list):
            hit_count += 1

    # 5. 输出结果
    print("" + "="*40)
    print(f"年份热门召回评估报告 (Independent)")
    print("="*40)
    print(f"召回规模 (K): {k}")
    print(f"时间窗口: [RefYear-2, RefYear+2]")
    print(f"有效命中数: {hit_count}")
    print(f"Hit Rate @ {k}: {hit_count / total_users:.4%}")
    print("="*40)

if __name__ == "__main__":
    run_year_popularity_recall(k=50)
