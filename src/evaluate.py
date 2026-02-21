import os
# --- 环境冲突修复 ---
# 解决 macOS 上多个 OpenMP 库竞争导致的崩溃问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pandas as pd
import numpy as np
import pickle
import faiss
import random
from tqdm import tqdm
from src.model import DualTowerModel

def get_mixed_history(full_seq, pos, recent_len=50, random_len=10):
    """
    构造混合历史特征：50个最近点击 + 10个远期随机点击。
    必须与训练逻辑严格对齐。
    """
    history_all = full_seq[:pos]
    if not history_all: 
        return [0] * (recent_len + random_len)
    
    recent_part = history_all[-recent_len:]
    remaining_part = history_all[:-recent_len]
    
    if len(remaining_part) > random_len:
        random_part = random.sample(remaining_part, random_len)
    else:
        random_part = remaining_part
        
    combined = recent_part + random_part
    max_len = recent_len + random_len
    if len(combined) < max_len:
        combined = combined + [0] * (max_len - len(combined))
    return combined

def evaluate(model_path, k_list=[10, 50, 100]):
    """
    全量召回评估：强制使用 CPU 以确保在 Mac M系列芯片上的稳定性。
    """
    # --- 稳定性优化：评估阶段回归 CPU ---
    # 理由：规避 MPS 加速下 Faiss 与 PyTorch 的底层线程冲突，消除 Segmentation Fault
    device = torch.device("cpu")
    # 限制 Faiss 使用单线程，进一步提升在多任务环境下的稳定性
    faiss.omp_set_num_threads(1)
    
    print(f"正在使用的评估设备: {device} (已开启稳定性保护)")

    # 1. 加载元数据
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # 2. 初始化模型并加载权重
    # map_location='cpu' 确保即使是在 MPS 上训练的模型也能在 CPU 上打开
    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    # 3. 生成全量电影向量池
    print("正在生成全量电影向量池...")
    item_count = meta['item_count']
    all_item_indices = torch.arange(item_count).to(device)
    
    max_genre_len = 6
    all_genres = []
    for i in range(item_count):
        g = meta.get('movie_genres', {}).get(i, [])
        if len(g) > max_genre_len: g = g[:max_genre_len]
        else: g = g + [0] * (max_genre_len - len(g))
        all_genres.append(g)
    all_genres_tensor = torch.tensor(all_genres).to(device)

    with torch.no_grad():
        item_embs_list = []
        batch_size = 4096 # CPU 内存通常比显存大，可以适当调大 batch
        for i in range(0, item_count, batch_size):
            embs = model.forward_item(all_item_indices[i : i + batch_size], 
                                    all_genres_tensor[i : i + batch_size])
            item_embs_list.append(embs.numpy().astype('float32'))
        item_pool_np = np.vstack(item_embs_list)

    # 4. 构建 Faiss 索引
    print(f"正在构建 Faiss 索引 (维度: {item_pool_np.shape[1]})...")
    index = faiss.IndexFlatIP(item_pool_np.shape[1])
    index.add(item_pool_np)

    # 5. 加载测试集
    print("正在加载测试数据...")
    test_df = pd.read_csv('data/processed/test.csv')
    user_sequences = meta['user_sequences']
    
    hits = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    total = len(test_df)

    # 6. 执行全量召回检索
    print(f"正在为 {total} 个测试用户进行 Top-K 检索评估 (Recall/NDCG)...")
    u_batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, total, u_batch_size)):
            batch_df = test_df.iloc[i : i + u_batch_size]
            
            u_indices = torch.tensor(batch_df['user_idx'].values).to(device)
            # 动态截取历史，严格无穿越
            u_hists = [get_mixed_history(user_sequences.get(u, []), pos) 
                       for u, pos in zip(batch_df['user_idx'], batch_df['pos'])]
            u_hists_tensor = torch.tensor(u_hists).to(device)
            
            # 计算用户向量
            u_vecs = model.forward_user(u_indices, u_hists_tensor).numpy().astype('float32')
            
            # 向量检索
            max_k = max(k_list)
            _, topk_indices = index.search(u_vecs, max_k)
            
            targets = batch_df['item_idx'].values
            for j in range(len(targets)):
                target = targets[j]
                user_topk = topk_indices[j]
                
                if target in user_topk:
                    rank = np.where(user_topk == target)[0][0]
                    for k in k_list:
                        if rank < k:
                            hits[k] += 1
                            ndcgs[k] += 1.0 / np.log2(rank + 2)

    # 7. 报告
    print("\n" + "="*40)
    print(f"召回评估报告 (正样本阈值: 3.0)")
    print("="*40)
    for k in k_list:
        hr = hits[k] / total
        ndcg = ndcgs[k] / total
        print(f"Top-{k:<3} | Hit Rate: {hr:.4f} | NDCG: {ndcg:.4f}")
    print("="*40)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "tower_model_epoch_5.pth"
    evaluate(model_file)
