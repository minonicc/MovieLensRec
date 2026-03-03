import torch
import pandas as pd
import numpy as np
import pickle
import faiss
import os
import random
from tqdm import tqdm
from ..model import DualTowerModel

# 解决 macOS 上由于多个库冲突导致的 OpenMP 重复初始化崩溃问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_mixed_history(full_seq, pos):
    """
    构造混合历史特征：50个最近点击 + 10个远期随机点击。
    """
    h = full_seq[:pos]
    return (h[-50:] + [0]*60)[:60]

def get_hist_genres(hist_ids, movie_genres):
    """
    展平用户历史题材并截断/填充至 100 维。
    """
    genres = []
    for i in hist_ids:
        if i != 0: genres.extend(movie_genres.get(i, []))
    return (genres[:100] + [0]*100)[:100]

def main():
    device = torch.device('cpu')
    # 使用用户指定的最佳全量模型
    model_path = '/Users/mino/Documents/workspace/movielens/best_tower_model_all.pth'
    
    print('正在加载元数据与测试集索引...')
    with open('data/processed/meta.pkl', 'rb') as f: 
        meta = pickle.load(f)
    test_df = pd.read_csv('data/processed/test.csv')
    
    print('统计训练集中电影的真实评分次数以定义冷门度...')
    item_counts = {}
    for seq in meta['user_sequences'].values():
        for i in seq: 
            item_counts[i] = item_counts.get(i, 0) + 1
            
    print('加载模型结构与权重 (支持跨版本加载)...')
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    # 使用 strict=False 保证即使特征维度有微调也能加载成功
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()
    
    print('正在生成全量电影向量池...')
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count).to(device)
    all_genres, all_stats = [], []
    for i in range(item_count):
        g = meta['movie_genres'].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6])
        f = meta['item_feat_map'].get(i, {})
        all_stats.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
        
    all_genres_t = torch.tensor(all_genres).to(device)
    all_stats_t = torch.tensor(all_stats).to(device)

    with torch.no_grad():
        item_embs = []
        for i in range(0, item_count, 4096):
            embs = model.forward_item(all_item_indices[i : i + 4096], 
                                    all_genres_t[i : i + 4096], 
                                    all_stats_t[i : i + 4096])
            item_embs.append(embs.cpu().numpy().astype('float32'))
        item_pool = np.vstack(item_embs)
        
    index = faiss.IndexFlatIP(item_pool.shape[1])
    index.add(item_pool)

    print('开始统计命中用户中的冷门分布 (Top-100)...')
    hit_cold_5, hit_cold_10, total_hits = 0, 0, 0
    movie_genres = meta['movie_genres']
    user_sequences = meta['user_sequences']
    user_feat_map = meta['user_feat_map']
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        u_idx, target, pos = int(row['user_idx']), int(row['item_idx']), int(row['pos'])
        u_hist = get_mixed_history(user_sequences.get(u_idx, []), pos)
        u_hgen = get_hist_genres(u_hist, movie_genres)
        f = user_feat_map.get(u_idx, {})
        u_stat = [f.get('rating_bucket', 0), f.get('count_bucket', 0)]
        
        with torch.no_grad():
            u_vec = model.forward_user(torch.tensor([u_idx]), torch.tensor([u_hist]), 
                                     torch.tensor([u_hgen]), torch.tensor([u_stat])).numpy().astype('float32')
        _, topk = index.search(u_vec, 100)
        
        if target in topk[0]:
            total_hits += 1
            cnt = item_counts.get(target, 0)
            if cnt < 5: hit_cold_5 += 1
            if cnt < 10: hit_cold_10 += 1

    print('' + '='*40)
    print(f'分析报告：双塔命中用户 (Top-100) 冷门度统计')
    print('='*40)
    print(f'测试集总命中数: {total_hits}')
    print(f'其中点击电影评分数 < 5 的命中数: {hit_cold_5}')
    print(f'其中点击电影评分数 < 10 的命中数: {hit_cold_10}')
    print('='*40)

if __name__ == '__main__':
    main()
