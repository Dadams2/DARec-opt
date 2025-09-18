from torch.utils.data import DataLoader, Dataset
from collections import Counter
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from AutoRec import *

class Mydata(Dataset):
    def __init__(self, S_path, T_path, train_ratio=0.9, test_ratio=0.1, train=None, preprocessed=True):
        super().__init__()

        self.S_path = S_path
        self.T_path = T_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        if preprocessed == False:
            # Check if output files already exist
            S_base = os.path.basename(self.S_path)
            T_base = os.path.basename(self.T_path)
            S_out = os.path.join(os.path.dirname(self.S_path), f'I_{S_base}_{T_base}.npy')
            T_out = os.path.join(os.path.dirname(self.T_path), f'I_{T_base}_{S_base}.npy')
            
            if os.path.exists(S_out) and os.path.exists(T_out):
                print(f"Output files already exist: {S_out}, {T_out}. Skipping preprocessing.")
                self.S_data = np.load(S_out)
                self.T_data = np.load(T_out)
            else:
                S_df = pd.read_csv(S_path, header=None)
                T_df = pd.read_csv(T_path, header=None)
                S_df.columns = ["User", "Item", "Rating", "TimeStamp"]
                T_df.columns = ["User", "Item", "Rating", "TimeStamp"]

                S_cnt = Counter(S_df.iloc[:, 0])
                S_cnt = {k: v for k, v in S_cnt.items() if v >= 5}
                T_cnt = Counter(T_df.iloc[:, 0])
                T_cnt = {k: v for k, v in T_cnt.items() if v >= 5}
                ### 求两个人名的交集
                S_user = set(S_cnt.keys())
                T_user = set(T_cnt.keys())
                users = list(S_user.intersection(T_user))
                S_df = S_df.loc[S_df["User"].isin(users)]
                T_df = T_df.loc[T_df["User"].isin(users)]
                ### 所有用户对应一个序号
                dict_users = {users[i]: i for i in range(len(users))}

                S_items = list(set(S_df.iloc[:, 1]))

                T_items = list(set(T_df.iloc[:, 1]))
                ### 所有数据对应一个序号
                # print(len(T_items))
                dict_S_items = {S_items[i]: i for i in range(len(S_items))}
                dict_T_items = {T_items[i]: i for i in range(len(T_items))}
                S_df.reset_index(drop=True, inplace=True)
                T_df.reset_index(drop=True, inplace=True)
                ### 将代码转换为序号
                for index, row in tqdm(S_df.iterrows()):
                    user_idx = dict_users[row["User"]]
                    item_idx = dict_S_items[row["Item"]]
                    S_df.iloc[index, 0] = user_idx
                    S_df.iloc[index, 1] = item_idx
                for index, row in tqdm(T_df.iterrows()):
                    user_idx = dict_users[row["User"]]
                    item_idx = dict_T_items[row["Item"]]
                    T_df.iloc[index, 0] = user_idx
                    T_df.iloc[index, 1] = item_idx
                # 样例
                # Item    User    Rating
                # 22      32          5
                # print(S_df.head())
                # (len(users))
                # 构建rating matrix
                self.S_data = torch.zeros((len(users), len(S_items)))
                self.T_data = torch.zeros((len(users), len(T_items)))
                for index, row in tqdm(S_df.iterrows()):
                    user = row["User"]
                    item = row["Item"]
                    self.S_data[user, item] = row["Rating"]

                for index, row in tqdm(T_df.iterrows()):
                    user = row["User"]
                    item = row["Item"]
                    self.T_data[user, item] = row["Rating"]

                np.save(S_out, self.S_data)
                np.save(T_out, self.T_data)
                print(f"Saved: {S_out}, {T_out}")
            
            self.S_out = S_out
            self.T_out = T_out
        else:
            S_base = os.path.basename(self.S_path)
            T_base = os.path.basename(self.T_path)
            S_out = os.path.join(os.path.dirname(self.S_path), f'I_{S_base}_{T_base}.npy')
            T_out = os.path.join(os.path.dirname(self.T_path), f'I_{T_base}_{S_base}.npy')
            self.S_data = np.load(S_out)
            self.T_data = np.load(T_out)

        self.S_data = self.S_data.T
        self.T_data = self.T_data.T

        if (self.S_data.shape[0] > self.T_data.shape[0]):
            diff = self.S_data.shape[0] - self.T_data.shape[0]
            self.T_data = np.concatenate((self.T_data, self.T_data[:diff]), axis=0)
        else:
            diff = self.T_data.shape[0] - self.S_data.shape[0]
            self.S_data = np.concatenate((self.S_data, self.S_data[:diff]), axis=0)
        np.random.seed(0)
        self.S_y = torch.zeros((self.S_data.shape[0], 1))
        self.T_y = torch.ones((self.T_data.shape[0], 1))
        
        # Ensure both datasets have the same number of samples after potential padding
        num_samples = min(self.S_data.shape[0], self.T_data.shape[0])
        self.total_indices = np.arange(num_samples)
        
        # Calculate test size and ensure it doesn't exceed available samples
        test_size = int(num_samples * self.test_ratio)
        test_size = min(test_size, num_samples)  # Ensure test_size doesn't exceed num_samples
        
        self.test_indices = np.random.choice(self.total_indices, size=test_size, replace=False)
        self.train_indices = np.array(list(set(self.total_indices) - set(self.test_indices)))


        if train != None:
            if train == True:
                self.S_data = self.S_data[self.train_indices]
                self.T_data = self.T_data[self.train_indices]
                self.S_y = self.S_y[self.train_indices]
                self.T_y = self.T_y[self.train_indices]
            else:
                self.S_data = self.S_data[self.test_indices]
                self.T_data = self.T_data[self.test_indices]
                self.S_y = self.S_y[self.test_indices]
                self.T_y = self.T_y[self.test_indices]
        print(self.S_data.shape)
        print(self.T_data.shape)

        # print(self.T_data.shape)
        # print(self.S_y.shape)
        # print(self.T_y.shape)
    def __getitem__(self, item):
        return (self.S_data[item], self.T_data[item], self.S_y[item], self.T_y[item])

    def __len__(self):
        return self.S_data.shape[0]
if __name__ == "__main__":
    DATA_DIR = "data"  # Change this to your data directory
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('ratings_') and f.endswith('.csv')]
    files.sort()
    for i in range(len(files)):
        for j in range(len(files)):
            if i == j:
                continue
            S_file = files[i]
            T_file = files[j]
            S_path = os.path.join(DATA_DIR, S_file)
            T_path = os.path.join(DATA_DIR, T_file)
            
            # Check if output files already exist
            S_base = os.path.basename(S_path)
            T_base = os.path.basename(T_path)
            S_out = os.path.join(os.path.dirname(S_path), f'I_{S_base}_{T_base}.npy')
            T_out = os.path.join(os.path.dirname(T_path), f'I_{T_base}_{S_base}.npy')
            
            if os.path.exists(S_out) and os.path.exists(T_out):
                print(f"Output files already exist for S: {S_file}, T: {T_file}. Skipping.")
                continue
                
            print(f"Processing S: {S_file}, T: {T_file}")
            data = Mydata(S_path, T_path, train=True, preprocessed=False)
            print(f"Processing completed for S: {S_file}, T: {T_file}")
