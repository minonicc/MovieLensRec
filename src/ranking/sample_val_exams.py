import pandas as pd
import os

def sample_validation_exams():
    """
    从全量验证集中精准抽样 1 万个用户，生成轻量级考卷。
    这能大幅提升 DeepFM 训练过程中每个 Epoch 的验证速度。
    """
    input_path = 'data/ranking_processed/val_exams.parquet'
    output_path = 'data/ranking_processed/val_exams_10k.parquet'
    
    if not os.path.exists(input_path):
        print(f"错误：未找到输入文件 {input_path}。")
        return

    print(f"正在从 {input_path} 加载全量验证数据...")
    df_val = pd.read_parquet(input_path)
    
    # 执行随机抽样
    # 设置随机种子保证多次实验的验证集是同一个
    sample_size = min(10000, len(df_val))
    df_val_sampled = df_val.sample(n=sample_size, random_state=42)
    
    print(f"抽样完成。已从 {len(df_val)} 用户中选取 {len(df_val_sampled)} 用户。")
    
    # 保存结果
    df_val_sampled.to_parquet(output_path, index=False)
    print(f">>> 精简版验证考卷已保存至: {output_path}")

if __name__ == "__main__":
    sample_validation_exams()
