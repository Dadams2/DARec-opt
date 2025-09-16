from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import torch
from AutoRec import *

class Mydata(Dataset):
    def __init__(self, S_path, T_path, train_ratio=0.9, test_ratio=0.1, train=None, preprocessed=True, mode='I'):
        super().__init__()
        self.S_path = S_path
        self.T_path = T_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.mode = mode
        if preprocessed == False:
            S_df = pd.read_csv(S_path, header=None)
            T_df = pd.read_csv(T_path, header=None)
            S_df.columns = ["User", "Item", "Rating", "TimeStamp"]
            T_df.columns = ["User", "Item", "Rating", "TimeStamp"]
            S_cnt = Counter(S_df.iloc[:, 0])
            S_cnt = {k: v for k, v in S_cnt.items() if v >= 5}
            T_cnt = Counter(T_df.iloc[:, 0])
            T_cnt = {k: v for k, v in T_cnt.items() if v >= 5}
            S_user = set(S_cnt.keys())
            T_user = set(T_cnt.keys())
            users = list(S_user.intersection(T_user))
            S_df = S_df.loc[S_df["User"].isin(users)]
            T_df = T_df.loc[T_df["User"].isin(users)]
            dict_users = {users[i]: i for i in range(len(users))}
            S_items = list(set(S_df.iloc[:, 1]))
            T_items = list(set(T_df.iloc[:, 1]))
            dict_S_items = {S_items[i]: i for i in range(len(S_items))}
            dict_T_items = {T_items[i]: i for i in range(len(T_items))}
            S_df.reset_index(drop=True, inplace=True)
            T_df.reset_index(drop=True, inplace=True)
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
            S_base = os.path.basename(self.S_path)
            T_base = os.path.basename(self.T_path)
            S_out = os.path.join(os.path.dirname(self.S_path), f'{self.mode}_' + S_base + '.npy')
            T_out = os.path.join(os.path.dirname(self.T_path), f'{self.mode}_' + T_base + '.npy')
            np.save(S_out, self.S_data)
            np.save(T_out, self.T_data)
            self.S_out = S_out
            self.T_out = T_out
        else:
            S_base = os.path.basename(self.S_path)
            T_base = os.path.basename(self.T_path)
            S_out = os.path.join(os.path.dirname(self.S_path), f'{self.mode}_' + S_base + '.npy')
            T_out = os.path.join(os.path.dirname(self.T_path), f'{self.mode}_' + T_base + '.npy')
            self.S_data = np.load(S_out)
            self.T_data = np.load(T_out)
        if self.mode == 'I':
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
        self.total_indices = np.arange(self.S_data.shape[0])
        self.test_indices = np.random.choice(self.total_indices, size=int(len(self.S_data) * self.test_ratio), replace=False)
        self.train_indices = np.array(list(set(self.total_indices)-set(self.test_indices)))
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
        print(f"{self.mode} mode shapes:", self.S_data.shape, self.T_data.shape)

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
            print(f"Processing S: {S_file}, T: {T_file}")
            # U-DARec mode
            data_U = Mydata(S_path, T_path, train=True, preprocessed=False, mode='U')
            print(f"Saved U: {data_U.S_out}, {data_U.T_out}")
            # I-DARec mode
            data_I = Mydata(S_path, T_path, train=True, preprocessed=False, mode='I')
            print(f"Saved I: {data_I.S_out}, {data_I.T_out}")
