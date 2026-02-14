import pandas as pd
import numpy as np
import pickle
import os

def preprocess(data_dir='data/ml-32m/', output_dir='data/processed/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    # 指定dtype减少内存占用
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int})

    # 1. 隐式反馈处理：仅保留 4分及以上作为正样本
    print("Filtering positive ratings...")
    pos_ratings = ratings[ratings['rating'] >= 4.0].copy()
    
    # 2. ID 映射 (UserId, MovieId -> Continuous Index)
    print("Mapping IDs...")
    user_list = pos_ratings['userId'].unique()
    user2id = {id: i for i, id in enumerate(user_list)}
    
    movie_list = movies['movieId'].unique()
    movie2id = {id: i for i, id in enumerate(movie_list)}
    
    pos_ratings['user_idx'] = pos_ratings['userId'].map(user2id)
    pos_ratings['item_idx'] = pos_ratings['movieId'].map(movie2id)
    
    # 3. 处理电影 Genres
    print("Processing genres...")
    all_genres = set()
    for g in movies['genres'].str.split('|'):
        all_genres.update(g)
    genre2id = {g: i+1 for i, g in enumerate(sorted(list(all_genres)))}
    
    movie_genres = {}
    for _, row in movies.iterrows():
        m_idx = movie2id.get(row['movieId'])
        if m_idx is not None:
            g_ids = [genre2id[g] for g in row['genres'].split('|') if g in genre2id]
            movie_genres[m_idx] = g_ids

    # 4. 按用户时间戳排序并生成序列 (Leave-one-out)
    print("Splitting data (Leave-one-out)...")
    pos_ratings.sort_values(['user_idx', 'timestamp'], ascending=[True, True], inplace=True)
    
    # 聚合每个用户的物品序列
    user_history_all = pos_ratings.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    train_data = []
    val_data = []
    test_data = []
    final_user_history = {} 

    for u_idx, items in user_history_all.items():
        if len(items) < 3: 
            continue
        
        # 最后一条测试，倒数第二条验证
        test_data.append([u_idx, items[-1]])
        val_data.append([u_idx, items[-2]])
        
        # 其余训练
        for i in items[:-2]:
            train_data.append([u_idx, i])
            
        # 记录训练集历史，用于模型输入特征
        final_user_history[u_idx] = items[:-2]

    print("Saving processed data...")
    meta = {
        'user_count': len(user2id),
        'item_count': len(movie2id),
        'genre_count': len(genre2id) + 1,
        'movie_genres': movie_genres,
        'user_history': final_user_history
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    pd.DataFrame(train_data, columns=['user_idx', 'item_idx']).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame(val_data, columns=['user_idx', 'item_idx']).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    pd.DataFrame(test_data, columns=['user_idx', 'item_idx']).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Preprocess finished! Train samples: {len(train_data)}")

if __name__ == "__main__":
    preprocess()
