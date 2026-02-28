import pandas as pd
import numpy as np
import pickle
import math
import os
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def evaluate_single_batch(batch_data, item_sim_matrix, user_sequences, k_list):
    """
    单个进程执行的任务：为一批用户计算 ItemCF 召回指标。
    """
    hits = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    max_k = max(k_list)
    
    for _, row in batch_data.iterrows():
        u_idx = int(row['user_idx'])
        target_item = int(row['item_idx'])
        pos = int(row['pos'])
        
        # 1. 获取该用户当前时刻之前的历史序列 (严格无穿越)
        history = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history)
        
        # 2. ItemCF 触发逻辑
        rank_list = defaultdict(float)
        for h_item in history:
            if h_item in item_sim_matrix:
                for neighbor, score in item_sim_matrix[h_item]:
                    if neighbor not in history_set:
                        rank_list[neighbor] += score
        
        if not rank_list: continue
        
        # 3. 排序获取 Top-K 召回结果 (按得分降序)
        sorted_recall = sorted(rank_list.items(), key=lambda x: x[1], reverse=True)
        topk_items = [x[0] for x in sorted_recall[:max_k]]
        
        # 4. 计算指标
        if target_item in topk_items:
            rank = topk_items.index(target_item)
            for k in k_list:
                if rank < k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / math.log2(rank + 2)
                    
    return hits, ndcgs

def evaluate_item_cf(k_list=[10, 50, 100]):
    """
    专门针对 ItemCF 召回路进行多进程并行评估。
    """
    print("正在加载元数据与相似度矩阵...")
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    with open('data/processed/item_cf_sim.pkl', 'rb') as f:
        item_sim_matrix = pickle.load(f)
        
    test_df = pd.read_csv('data/processed/test.csv')
    user_sequences = meta['user_sequences']
    
    total_users = len(test_df)
    # 根据 CPU 核心数决定进程数，通常留 1-2 个核心给系统
    num_workers = max(1, os.cpu_count() - 1)
    print(f"正在启动 {num_workers} 个进程进行并行评估 (总测试用户: {total_users})...")

    # 将测试集切分为多个小块
    batches = np.array_split(test_df, num_workers)
    
    # 存储所有进程返回的结果
    all_hits = {k: 0 for k in k_list}
    all_ndcgs = {k: 0.0 for k in k_list}

    # 使用进程池执行并行任务
    # 注意：item_sim_matrix 和 user_sequences 作为参数传递时，
    # 在 Unix/macOS 系统下会利用 Fork 机制实现高效的只读共享。
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_single_batch, batch, item_sim_matrix, user_sequences, k_list) 
                   for batch in batches]
        
        # 使用 tqdm 监控任务完成情况
        for future in tqdm(futures, desc="并行评估进度"):
            batch_hits, batch_ndcgs = future.result()
            for k in k_list:
                all_hits[k] += batch_hits[k]
                all_ndcgs[k] += batch_ndcgs[k]

    # 输出最终结果报告
    print("\n" + "="*40)
    print("ItemCF 召回路并行评估报告 (多进程加速)")
    print("="*40)
    for k in k_list:
        hr = all_hits[k] / total_users
        ndcg = all_ndcgs[k] / total_users
        print(f"Top-{k:<3} | Hit Rate: {hr:.4f} | NDCG: {ndcg:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_item_cf()
