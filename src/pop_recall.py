import pandas as pd
import numpy as np
import pickle
import os
import re
from tqdm import tqdm
from collections import defaultdict

def run_popularity_recall():
    """
    执行热门召回评估：
    1. 全局 Top-50 热门
    2. 年份窗口 Top-50 热门 (用户最后点击年份附近)
    分别统计指标并计算去重后的总命中率。
    """
    print("正在加载元数据与评分数据...")
    with open('data/processed/meta.pkl', 'rb') as f: meta = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # 1. 基础预处理：提取年份
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies['year'] = movies['title'].apply(extract_year)
    
    movie2id = {id: i+1 for i, id in enumerate(movies['movieId'].unique())}
    movies['item_idx'] = movies['movieId'].map(movie2id)
    id2year = movies.set_index('item_idx')['year'].to_dict()

    # 2. 统计电影流行度 (按训练集中的评分次数)
    print("计算电影流行度分布...")
    item_counts = defaultdict(int)
    for seq in meta['user_sequences'].values():
        for item_idx in seq: item_counts[item_idx] += 1
    
    movies['popularity'] = movies['item_idx'].map(item_counts).fillna(0)

    # 3. 预处理热门候选集
    # A. 全局 Top-200 (多备一点，方便去重和过滤已看)
    global_hot_list = movies.sort_values('popularity', ascending=False)['item_idx'].tolist()[:200]
    
    # B. 按年份预热候选池
    year_hot_map = {}
    for year in movies['year'].unique():
        if year == 0: continue
        year_hot_map[year] = movies[movies['year'] == year].sort_values('popularity', ascending=False)['item_idx'].tolist()[:100]

    # 4. 执行召回评估
    user_sequences = meta['user_sequences']
    total_users = len(test_df)
    
    hit_global = 0    # 仅全局路命中
    hit_year = 0      # 仅年份路命中
    hit_both = 0      # 两路均命中
    hit_total = 0     # 总命中
    
    print(f"正在对 {total_users} 个用户进行热门召回分析 (50全局 + 50年份)...")
    
    for _, row in tqdm(test_df.iterrows(), total=total_users):
        u_idx = int(row['user_idx'])
        target = int(row['item_idx'])
        pos = int(row['pos'])
        
        # 获取用户训练集历史用于过滤
        history = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history)
        
        # 确定参考年份 (最后一次点击的年份)
        last_item = history[-1] if history else 0
        ref_year = id2year.get(last_item, 2000)
        
        # --- 路 A: 全局热门召回 (Top-50) ---
        recall_global = []
        for m_idx in global_hot_list:
            if m_idx not in history_set:
                recall_global.append(m_idx)
                if len(recall_global) >= 50: break
        
        # --- 路 B: 年份热门召回 (Top-50) ---
        # 窗口：[ref_year-1, ref_year, ref_year+1]
        year_candidates = []
        for y in [ref_year, ref_year-1, ref_year+1, ref_year-2, ref_year+2]: # 扩大年份窗口增加候选多样性
            year_candidates.extend(year_hot_map.get(y, []))
        
        # 按热度重新排序并取 Top-50
        year_candidates = sorted(list(set(year_candidates)), key=lambda x: item_counts.get(x, 0), reverse=True)
        
        recall_year = []
        for m_idx in year_candidates:
            if m_idx not in history_set:
                recall_year.append(m_idx)
                if len(recall_year) >= 50: break
        
        # --- 指标统计 ---
        in_global = target in recall_global
        in_year = target in recall_year
        
        if in_global and in_year: hit_both += 1
        elif in_global: hit_global += 1
        elif in_year: hit_year += 1
        
        if in_global or in_year: hit_total += 1

    # 5. 输出报告
    print("" + "="*50)
    print("热门召回组件化评估报告 (50 Global + 50 Year-Context)")
    print("="*50)
    print(f"有效测试样本数: {total_users}")
    print(f"全局热门路命中数: {hit_global + hit_both} (HR: {(hit_global + hit_both)/total_users:.2%})")
    print(f"年份热门路命中数: {hit_year + hit_both} (HR: {(hit_year + hit_both)/total_users:.2%})")
    print("-" * 30)
    print(f"两路同时命中数: {hit_both}")
    print(f"两路去重后总命中: {hit_total} (Total HR: {hit_total/total_users:.2%})")
    print("="*50)
    print("分析结论：")
    print(f"年份路带来的『纯增量』命中: {hit_year} 人")
    print(f"热门召回对整体的贡献上限约为: {hit_total/total_users:.2%}")

if __name__ == "__main__":
    run_popularity_recall()
