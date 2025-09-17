from torch import nn, optim
from AutoRec import *
from Data_Preprocessing import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import datetime
import json
import itertools
from pathlib import Path

class ExperimentConfig:
    def __init__(self, 
                 epochs=50,
                 batch_size=64,
                 lr=1e-3,
                 wd=1e-4,
                 n_factors=200,
                 output_dir="./experiment_results",
                 log_dir="./logs",
                 save_models=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.n_factors = n_factors
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.save_models = save_models
        
    def to_dict(self):
        return self.__dict__


train_dataset = Mydata("/home2/dadams/DARec-opt/data/ratings_Amazon_Instant_Video.csv", "/home2/dadams/DARec-opt/data/ratings_Electronics.csv", train=True, preprocessed=True)
test_dataset = Mydata("/home2/dadams/DARec-opt/data/ratings_Amazon_Instant_Video.csv", "/home2/dadams/DARec-opt/data/ratings_Electronics.csv", train=False, preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

print("Data is loaded")
if args.train_S == True:
    n_items, n_users = train_dataset.S_data.shape[0], train_dataset.S_data.shape[1]
else:
    n_items, n_users = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]

model = I_AutoRec(n_users=n_users, n_items=n_items, n_factors=args.n_factors).cuda()
criterion = MRMSELoss().cuda()

optimizer = optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)

def train(epoch):
    model.train()
    Total_RMSE = 0
    Total_MASK = 0
    loc = 0 if args.train_S == True else 1
    for idx, d in enumerate(train_loader):
        data = d[loc].cuda()
        optimizer.zero_grad()
        _, pred = model(data)
        pred.cuda()
        loss, mask = criterion(pred, data)
        Total_RMSE += loss.item()
        Total_MASK += torch.sum(mask).item()
        # RMSE = torch.sqrt(loss.item() / torch.sum(mask))
        loss.backward()
        optimizer.step()

    return math.sqrt(Total_RMSE / Total_MASK)



def test():
    model.eval()
    Total_RMSE = 0
    Total_MASK = 0
    loc = 0 if args.train_S == True else 1
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            data = d[loc].cuda()
            _, pred = model(data)
            pred.cuda()
            loss, mask = criterion(pred, data)
            Total_RMSE += loss.item()
            Total_MASK += torch.sum(mask).item()

    return math.sqrt(Total_RMSE / Total_MASK)

if __name__=="__main__":
    train_rmse = []
    test_rmse = []
    # wdir = r"D:\DARec_I\Pretrained_Parameters"
    wdir = r"Pretrained_Parameters"
    model_name = r'S_AutoRec' if args.train_S == True else r'T_AutoRec'
    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(test())
        if epoch % args.epochs == args.epochs - 1:
            torch.save(model.state_dict(), wdir + model_name + "_%d.pkl" % (epoch+1))
    print(min(test_rmse))
    plt.plot(range(args.epochs), train_rmse, range(args.epochs), test_rmse)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.xticks(range(0, args.epochs, 2))
    plt.show()
