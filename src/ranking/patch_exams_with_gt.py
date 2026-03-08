import pandas as pd
import os

def generate_capability_exams(src_name, dest_name):
    src_path = os.path.join('data/ranking_processed/', src_name)
    dest_path = os.path.join('data/ranking_processed/', dest_name)
    
    if not os.path.exists(src_path):
        print(f"跳过：未找到源文件 {src_path}")
        return

    print(f"正在从 {src_name} 生成排序能力版考卷 {dest_name}...")
    df = pd.read_parquet(src_path)
    
    def force_gt_in(row):
        gt = row['item_idx']
        cands = list(row['candidates'])
        if gt not in cands:
            if len(cands) >= 350: cands[349] = gt
            else: cands.append(gt)
        return cands

    df['candidates'] = df.apply(force_gt_in, axis=1)
    df.to_parquet(dest_path, index=False)
    print(f">>> {dest_name} 生成完成！")

if __name__ == "__main__":
    # 从端到端版生成排序能力版
    generate_capability_exams('val_exams_e2e.parquet', 'val_exams.parquet')
    generate_capability_exams('test_exams_e2e.parquet', 'test_exams_sampled_0.2.parquet')
    # 顺便处理一下 10k 验证集 (假设它目前也是 e2e 版)
    if os.path.exists('data/ranking_processed/val_exams_10k.parquet'):
        generate_capability_exams('val_exams_10k.parquet', 'val_exams_10k.parquet')
