import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pickle
import os
import faiss
import numpy as np
import pandas as pd
import random

# --- 全局配置 ---
# 温度系数：适配余弦相似度 [-1, 1] 范围，通过缩小分母放大相似度差异
TEMPERATURE = 0.07  
# 全局随机负样本数：Batch 内所有用户共享这 N 个负样本，性能极佳
GLOBAL_NEG_RATIO = 2048 

# 解决 macOS 上由于多个库冲突导致的 OpenMP 重复初始化崩溃问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" 

from src.dataset import MovieLensDataset, train_collate_fn 
from src.model import DualTowerModel


def evaluate_fast(model, meta, device, k=50):
    """
    在训练过程中快速计算验证集的 Hit Rate，用于早停判断。
    """
    model.eval()
    eval_device = torch.device("cpu")
    model.to(eval_device)
    faiss.omp_set_num_threads(1)

    # 1. 导出全量电影向量池
    print("正在生成全量电影向量池...")
    item_count = meta["item_count"]
    all_item_indices = torch.arange(item_count).to(eval_device)
    
    all_genres = []
    all_stats = []
    for i in range(item_count):
        g = meta["movie_genres"].get(i, [])
        all_genres.append((g[:6] + [0]*6)[:6]) 
        f = meta["item_feat_map"].get(i, {})
        all_stats.append([f.get("year_bucket", 0), f.get("rating_bucket", 0), f.get("count_bucket", 0)])
    
    all_genres_t = torch.tensor(all_genres).to(eval_device)
    all_stats_t = torch.tensor(all_stats).to(eval_device)

    with torch.no_grad():
        item_embs = []
        for i in range(0, item_count, 4096):
            embs = model.forward_item(all_item_indices[i:i+4096], all_genres_t[i:i+4096], all_stats_t[i:i+4096])
            item_embs.append(embs.numpy().astype("float32"))
        item_pool = np.vstack(item_embs)

    index = faiss.IndexFlatIP(item_pool.shape[1])
    index.add(item_pool)

    # 2. 加载验证集数据
    val_df = pd.read_csv("data/processed/val.csv")
    user_sequences = meta["user_sequences"]
    user_feat_map = meta["user_feat_map"]
    movie_genres = meta["movie_genres"]
    
    hit_count = 0
    total = len(val_df)
    
    u_batch_size = 2048 
    with torch.no_grad():
        for i in range(0, total, u_batch_size):
            batch_df = val_df.iloc[i : i + u_batch_size]
            u_ids = torch.tensor(batch_df["user_idx"].values).to(eval_device)
            
            u_hists = []
            u_hgenres = []
            u_stats = []
            for u, pos in zip(batch_df["user_idx"], batch_df["pos"]):
                full_seq = user_sequences.get(u, [])[:pos]
                recent = full_seq[-50:]
                rem = full_seq[:-50]
                rand = random.sample(rem, 10) if len(rem) > 10 else rem
                h = (recent + rand + [0]*60)[:60]
                u_hists.append(h)
                
                hg = []
                for item_id in h: 
                    if item_id != 0: hg.extend(movie_genres.get(item_id, []))
                u_hgenres.append((hg[:100] + [0]*100)[:100])

                f = user_feat_map.get(u, {})
                u_stats.append([f.get("rating_bucket", 0), f.get("count_bucket", 0)])
            
            u_vecs = model.forward_user(u_ids, torch.tensor(u_hists), torch.tensor(u_hgenres), torch.tensor(u_stats)).numpy().astype("float32")
            _, topk_indices = index.search(u_vecs, k)
            
            targets = batch_df["item_idx"].values
            for j in range(len(targets)):
                if targets[j] in topk_indices[j]: hit_count += 1

    model.train() 
    model.to(device) 
    return hit_count / total


def train():
    """
    模型训练主循环：实现基于『共享负采样』和『究极屏蔽』的高性能 InfoNCE 训练。
    """
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"正在使用的训练设备: {device}")

    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    # 预处理电影特征张量，存储在内存/显存中以提速
    item_count = meta['item_count']
    movie_genres = meta['movie_genres']
    item_feat_map = meta['item_feat_map']
    
    all_genres_mat = []
    all_stats_mat = []
    for i in range(item_count):
        g = movie_genres.get(i, [])
        all_genres_mat.append((g[:6] + [0]*6)[:6])
        f = item_feat_map.get(i, {})
        all_stats_mat.append([f.get('year_bucket', 0), f.get('rating_bucket', 0), f.get('count_bucket', 0)])
    
    all_genres_tensor = torch.tensor(all_genres_mat).to(device)
    all_stats_tensor = torch.tensor(all_stats_mat).to(device)

    train_dataset = MovieLensDataset('data/processed/train.csv', meta, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_collate_fn) 

    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_hr = 0.0
    patience = 3
    wait = 0

    print(f"开始高性能训练 (负采样强度: {GLOBAL_NEG_RATIO} 共享)...")
    for epoch in range(20): 
        start_time = time.time()
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, (u_ids, u_hists, u_hgenres, u_stats, t_i_ids, t_i_genres, t_i_stats) in pbar:
            # 移动数据到训练设备
            u_ids, u_hists, u_hgenres, u_stats = u_ids.to(device), u_hists.to(device), u_hgenres.to(device), u_stats.to(device)
            t_i_ids, t_i_genres, t_i_stats = t_i_ids.to(device), t_i_genres.to(device), t_i_stats.to(device)

            optimizer.zero_grad()
            
            # --- 步骤 1: 计算用户向量与正样本物品向量 ---
            user_embeddings = model.forward_user(u_ids, u_hists, u_hgenres, u_stats) # [Batch_Size, Embed_Dim]
            positive_item_embeddings = model.forward_item(t_i_ids, t_i_genres, t_i_stats) # [Batch_Size, Embed_Dim]
            
            # --- 步骤 2: 构造所有候选物品池 (正样本 + 共享负样本) ---
            # 随机生成共享负样本 ID
            neg_indices = torch.randint(1, item_count, (GLOBAL_NEG_RATIO,)).to(device)
            # 计算共享负样本 Embedding
            negative_item_embeddings = model.forward_item(neg_indices, all_genres_tensor[neg_indices], all_stats_tensor[neg_indices]) # [GLOBAL_NEG_RATIO, Embed_Dim]
            
            # 拼接正样本和负样本，构建统一候选池 [Batch + GLOBAL_NEG_RATIO, Embed_Dim]
            all_item_embs = torch.cat([positive_item_embeddings, negative_item_embeddings], dim=0)
            # 汇总所有候选物品的原始 ID，用于后续构造屏蔽 Mask
            all_candidate_ids = torch.cat([t_i_ids, neg_indices], dim=0)

            # --- 步骤 3: 计算所有 (用户, 候选物品) 的相似度 (Logits) ---
            # logits 形状: [Batch_Size, Batch_Size + GLOBAL_NEG_RATIO]
            logits = torch.matmul(user_embeddings, all_item_embs.T) / TEMPERATURE

            # --- 步骤 4: 构造『究极 Mask』屏蔽冲突项 ---
            # 4.1 历史碰撞屏蔽：如果候选物品出现在用户的 60 个特征历史中，则排除。
            mask_hist = (all_candidate_ids.view(1, -1) == u_hists.unsqueeze(2)).any(dim=1)
            # 4.2 目标碰撞屏蔽：如果候选物品撞上了本 Batch 的正样本 ID，则排除。
            mask_target = (all_candidate_ids.view(1, -1) == t_i_ids.view(-1, 1))
            # 4.3 同用户屏蔽：如果 Batch 内同一用户出现多次，则互相屏蔽。
            mask_user = torch.cat([
                (u_ids.unsqueeze(1) == u_ids.unsqueeze(0)), 
                torch.zeros((u_ids.size(0), GLOBAL_NEG_RATIO), dtype=torch.bool, device=device)
            ], dim=1)
            
            # 合并 Mask 并确保对角线上的正样本被保留 (fill_diagonal_(False))
            full_mask = (mask_hist | mask_target | mask_user).fill_diagonal_(False)

            # --- 步骤 5: 计算 InfoNCE Loss (Listwise 交叉熵) ---
            # 正样本在 logits 中的位置即为对角线索引
            labels = torch.arange(user_embeddings.size(0), dtype=torch.long, device=device)
            # 使用极小值覆盖冲突项，使其在 Softmax 之后贡献为 0
            loss = F.cross_entropy(logits.masked_fill(full_mask, -1e9), labels)
            
            # --- 反向传播与优化 ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- Epoch 统计与早停逻辑 ---
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束。平均 Loss: {avg_loss:.4f} | 耗时: {epoch_time:.2f}s")

        current_hr = evaluate_fast(model, meta, device, k=50)
        print(f"Epoch {epoch+1} 验证集 HitRate@50: {current_hr:.4f}")

        if current_hr > best_hr:
            best_hr = current_hr
            wait = 0
            torch.save(model.state_dict(), "best_tower_model.pth")
            print("发现更好的模型，已保存权重至 best_tower_model.pth")
        else:
            wait += 1
            print(f"验证集表现未提升，Patience: {wait}/{patience}")
            if wait >= patience:
                print("满足早停条件，停止训练。")
                break

if __name__ == "__main__":
    train()
