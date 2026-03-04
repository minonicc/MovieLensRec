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
# InfoNCE Loss 的温度系数，用于拉伸余弦相似度的分布，使模型学习更具区分度。
TEMPERATURE = 0.07  
# 全局随机负样本数：Batch 内所有用户共享这 N 个负样本，性能极佳。
GLOBAL_NEG_RATIO = 2048 

# 解决 macOS 上由于多个库冲突导致的 OpenMP 重复初始化崩溃问题。
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" 

from .dataset import MovieLensDataset, train_collate_fn 
from .model import DualTowerModel


def evaluate_fast(model, meta, device, k=50):
    """
    在训练过程中快速计算验证集的 Hit Rate，用于早停判断。
    该函数在 CPU 上运行以确保 Mac M系列芯片的稳定性。
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
            
            u_hists, u_hgenres, u_stats = [], [], []
            for u, pos in zip(batch_df["user_idx"], batch_df["pos"]):
                # 获取 50最近 + 10随机 的混合历史
                full_seq = user_sequences.get(u, [])[:pos]
                recent = full_seq[-50:]; rem = full_seq[:-50]
                rand = random.sample(rem, 10) if len(rem) > 10 else rem
                h = (recent + rand + [0]*60)[:60]
                u_hists.append(h)
                
                # 题材特征展平
                hg = []
                for item_id in h: 
                    if item_id != 0: hg.extend(movie_genres.get(item_id, []))
                u_hgenres.append((hg[:100] + [0]*100)[:100])

                # 用户侧统计特征
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
    模型训练主循环：实现集成『共享负采样』和『究极屏蔽』的高性能 InfoNCE 训练。
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

    # 加载元数据
    with open('data/processed/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    # 预处理电影特征张量，存储在内存/显存中以提速。
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
    
    # 转换为 Tensor 存储在显存中，用于极速负采样查询
    all_genres_tensor = torch.tensor(all_genres_mat).to(device)
    all_stats_tensor = torch.tensor(all_stats_mat).to(device)

    train_dataset = MovieLensDataset('data/processed/train.csv', meta, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_collate_fn) 

    model = DualTowerModel(meta['user_count'], meta['item_count'], meta['genre_count']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # --- 早停机制初始化 ---
    best_hr = 0.0
    patience = 3
    wait = 0

    print(f"开始高性能训练 (负采样强度: {GLOBAL_NEG_RATIO} 共享)...")
    for epoch in range(20): 
        start_time = time.time()
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in pbar:
            # 数据解包：4个User侧特征, 3个正样本Item特征, 3个专属困难样本特征
            u_ids, u_hists, u_hgenres, u_stats, t_ids, t_genres, t_stats, h_ids, h_genres, h_stats = [t.to(device) for t in batch]

            optimizer.zero_grad()
            
            # --- 步骤 1: 计算用户向量、正样本物品向量、专属困难负样本向量 ---
            user_embeddings = model.forward_user(u_ids, u_hists, u_hgenres, u_stats) # [Batch_Size, Embed_Dim]
            positive_embs = model.forward_item(t_ids, t_genres, t_stats)             # [Batch_Size, Embed_Dim]
            hard_negative_embs = model.forward_item(h_ids, h_genres, h_stats)       # [Batch_Size, Embed_Dim]
            
            # --- 步骤 2: 生成全局共享随机负样本 (Shared Global Negatives) ---
            # 这里的优化极大降低了 Item 塔的推理次数，提速显著。
            neg_indices = torch.randint(1, item_count, (GLOBAL_NEG_RATIO,)).to(device)
            random_negative_embs = model.forward_item(neg_indices, all_genres_tensor[neg_indices], all_stats_tensor[neg_indices]) # [GLOBAL_NEG_RATIO, Embed_Dim]
            
            # --- 步骤 3: 构造完整物品池 ---
            # 物品池顺序：[正样本(Batch_Size), 专属困难负项(Batch_Size), 共享随机负项(GLOBAL_NEG_RATIO)]
            all_items = torch.cat([positive_embs, hard_negative_embs, random_negative_embs], dim=0)
            all_item_ids = torch.cat([t_ids, h_ids, neg_indices], dim=0)

            # --- 步骤 4: 计算内积得分 (Logits) 并应用温度系数 ---
            # logits 形状: [Batch_Size, Batch_Size + Batch_Size + GLOBAL_NEG_RATIO]
            logits = torch.matmul(user_embeddings, all_items.T) / TEMPERATURE

            # --- 步骤 5: 构造『隔离对比与究极屏蔽』Mask ---
            # 5.1 正样本块屏蔽：排除同用户或同物品 ID 的干扰
            pos_mask = (u_ids.unsqueeze(1) == u_ids.unsqueeze(0)) | (t_ids.unsqueeze(1) == t_ids.unsqueeze(0))
            
            # 5.2 专属困难块屏蔽：每个用户仅与其对应的一行专属差评 H 对比，屏蔽他人的 H。
            hard_neg_mask = ~torch.eye(u_ids.size(0), dtype=torch.bool, device=device)
            # 同时屏蔽掉如果专属困难物品恰好出现在该用户历史特征中的情况
            h_hist_mask = (h_ids.view(1, -1) == u_hists.unsqueeze(2)).any(dim=1)
            hard_neg_mask = hard_neg_mask | h_hist_mask
            
            # 5.3 共享随机块屏蔽：屏蔽出现在特征历史或正样本目标中的随机负项
            rand_mask = (neg_indices.view(1, -1) == u_hists.unsqueeze(2)).any(dim=1) | (neg_indices.view(1, -1) == t_ids.unsqueeze(1))
            
            # 合并总掩码，并确保对角线上的正样本通过 fill_diagonal_(False) 被正确保留
            full_mask = torch.cat([pos_mask, hard_neg_mask, rand_mask], dim=1).fill_diagonal_(False)

            # --- 步骤 6: 计算 InfoNCE Loss (Listwise 交叉熵) ---
            # 标签设为 Batch 内的对角线索引。
            labels = torch.arange(u_ids.size(0), dtype=torch.long, device=device)
            # 使用 masked_fill 将冲突项设为极小值 (-1e9)，使其在 Softmax 分母中不产生贡献
            loss = F.cross_entropy(logits.masked_fill(full_mask, -1e9), labels)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- Epoch 统计与评估 ---
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束。平均 Loss: {avg_loss:.4f} | 耗时: {epoch_time:.2f}s")

        # 快速验证召回率
        current_hr = evaluate_fast(model, meta, device, k=50)
        print(f"Epoch {epoch+1} 验证集 HitRate@50: {current_hr:.4f}")

        # 早停逻辑：如果验证集 Hit Rate 连续 3 代不再增长，则停止
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
