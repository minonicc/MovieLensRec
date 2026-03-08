import pickle
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# --- 路径配置 ---
POOLS_PATH = 'data/ranking_processed/multi_snapshot_pools.pkl'
USER_PACK_PATH = 'data/ranking_processed/ranking_user_pack.pkl'
OUTPUT_DIR = 'data/ranking_processed/'

def split_and_prepare_exams():
    """
    严谨拆分 2.3G 巨型快照池。
    确保：
    1. 训练池只包含 [0, n-3] 的索引。
    2. 验证考卷只包含 n-2 的索引。
    3. 测试考卷只包含 n-1 的索引。
    """
    print(f"正在加载巨型文件 {POOLS_PATH} (2.3G)...")
    with open(POOLS_PATH, 'rb') as f:
        all_pools = pickle.load(f)
    
    print("加载黄金用户包...")
    with open(USER_PACK_PATH, 'rb') as f:
        user_pack = pickle.load(f)
    
    u_seqs = user_pack['user_sequences']
    ranking_uids = user_pack['user_ids']
    
    val_rows = []
    test_rows = []
    train_pools_only = {}

    print("执行严谨时序拆分与考卷封装...")
    for u in tqdm(ranking_uids):
        user_snaps = all_pools.get(u, {})
        if not user_snaps: continue
        
        n = len(u_seqs[u])
        if n < 3: continue
        
        # 1. 提取验证集考卷 (n-2)
        val_pos = n - 2
        if val_pos in user_snaps:
            val_rows.append({
                'user_idx': u,
                'item_idx': u_seqs[u][val_pos],
                'pos': val_pos,
                'candidates': user_snaps[val_pos]
            })
            
        # 2. 提取测试集考卷 (n-1)
        test_pos = n - 1
        if test_pos in user_snaps:
            test_rows.append({
                'user_idx': u,
                'item_idx': u_seqs[u][test_pos],
                'pos': test_pos,
                'candidates': user_snaps[test_pos]
            })
            
        # 3. 提取纯净训练池 (所有 <= n-3 的点)
        clean_train_snaps = {pos: cands for pos, cands in user_snaps.items() if pos <= n - 3}
        if clean_train_snaps:
            train_pools_only[u] = clean_train_snaps

    # --- 数据持久化 ---
    print("\n正在持久化轻量级考卷 (Parquet)...")
    pd.DataFrame(val_rows).to_parquet(os.path.join(OUTPUT_DIR, 'val_exams_e2e.parquet'))
    pd.DataFrame(test_rows).to_parquet(os.path.join(OUTPUT_DIR, 'test_exams_e2e.parquet'))
    
    print("正在持久化精简训练池 (Pickle)...")
    with open(os.path.join(OUTPUT_DIR, 'training_pools_only.pkl'), 'wb') as f:
        pickle.dump(train_pools_only, f)
        
    print(f"\n拆分圆满完成！")
    print(f"训练池规模: {len(train_pools_only)} 用户")
    print(f"验证考卷规模: {len(val_rows)} 用户")
    print(f"测试考卷规模: {len(test_rows)} 用户")
    print(">>> 现在可以安全删除原始的 2.3G 巨型文件了。")

if __name__ == "__main__":
    split_and_prepare_exams()
