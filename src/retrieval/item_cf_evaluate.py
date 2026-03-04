import pandas as pd
import numpy as np
import pickle
import math
import os
import random
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def evaluate_single_batch(batch_data, item_sim_matrix, user_sequences, k_list):
    """
    单个进程执行的任务：为一批用户计算 ItemCF 召回指标。
    已加入逐用户随机种子和稳定排序逻辑，确保结果绝对可复现。
    """
    hits = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    max_k = max(k_list)
    
    for _, row in batch_data.iterrows():
        u_idx = int(row['user_idx'])
        target_item = int(row['item_idx'])
        pos = int(row['pos'])
        
        # --- 核心改进：设置逐用户随机种子 ---
        # 确保每个用户在多次运行中的随机特征构造（如有）和排序表现完全一致
        random.seed(u_idx)
        np.random.seed(u_idx)
        
        # 1. 获取该用户当前时刻之前的历史序列 (严格遵循时间线，消除穿越)
        history = user_sequences.get(u_idx, [])[:pos]
        history_set = set(history)
        
        # 2. ItemCF 触发逻辑：遍历历史中的每部电影，寻找其预计算好的相似邻居
        rank_list = defaultdict(float)
        for h_item in history:
            if h_item in item_sim_matrix:
                for neighbor, score in item_sim_matrix[h_item]:
                    # 排除掉用户已经在训练集里看过的电影
                    if neighbor not in history_set:
                        rank_list[neighbor] += score
        
        if not rank_list: continue
        
        # 3. 排序获取 Top-K 召回结果
        # 优化点：使用 (-score, item_id) 进行双重排序，确保在相似度分值相等时，结果依然唯一确定
        sorted_recall = sorted(rank_list.items(), key=lambda x: (-x[1], x[0]))
        topk_items = [x[0] for x in sorted_recall[:max_k]]
        
        # 4. 计算召回指标：Hit Rate 和 NDCG
        if target_item in topk_items:
            # 找到目标物品在推荐列表中的具体排名 (0-based)
            rank = topk_items.index(target_item)
            for k in k_list:
                if rank < k:
                    hits[k] += 1
                    # NDCG@K = 1 / log2(rank + 2)
                    ndcgs[k] += 1.0 / math.log2(rank + 2)
                    
    return hits, ndcgs

def evaluate_item_cf(k_list=[10, 50, 100, 200, 300, 400, 500]):
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
    # 根据 CPU 核心数决定并行进程数，压榨多核性能
    num_workers = max(1, os.cpu_count() - 1)
    print(f"正在启动 {num_workers} 个进程进行并行评估 (总测试用户: {total_users})...")

    # 将测试集均匀切分为多个小块，分发给不同进程
    batches = np.array_split(test_df, num_workers)
    
    # 初始化统计容器
    all_hits = {k: 0 for k in k_list}
    all_ndcgs = {k: 0.0 for k in k_list}

    # 使用进程池执行任务：子进程通过 fork 机制共享内存中的大型相似度矩阵
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_single_batch, batch, item_sim_matrix, user_sequences, k_list) 
                   for batch in batches]
        
        # 使用 tqdm 实时监控并行的进度
        for future in tqdm(futures, desc="并行评估进度"):
            batch_hits, batch_ndcgs = future.result()
            for k in k_list:
                all_hits[k] += batch_hits[k]
                all_ndcgs[k] += batch_ndcgs[k]

    # 输出最终结果报告
    print("\n" + "="*40)
    print("ItemCF 召回路并行评估报告 (可复现版)")
    print("="*40)
    for k in k_list:
        hr = all_hits[k] / total_users
        ndcg = all_ndcgs[k] / total_users
        print(f"Top-{k:<3} | Hit Rate: {hr:.4f} | NDCG: {ndcg:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_item_cf()
