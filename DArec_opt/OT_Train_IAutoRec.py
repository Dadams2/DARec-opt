import torch
from torch import nn, optim
from DArec_opt.OT_IAutoRec import OT_AutoRec
from Data_Preprocessing import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def check_positive(val):
    val = int(val)
    if val <=0:
        raise argparse.ArgumentError(f'{val} is invalid value. epochs should be positive integer')
    return val

parser = argparse.ArgumentParser(description='OT-AutoRec with PyTorch')
parser.add_argument('--epochs', '-e', type=check_positive, default=50)
parser.add_argument('--batch_size', '-b', type=check_positive , default=64)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-4)
parser.add_argument('--n_factors', type=int, help="embedding size of autoencoder", default=200)
parser.add_argument('--ot_weight', type=float, help="weight for OT loss", default=0.1)
args = parser.parse_args()

train_dataset = Mydata("/home2/dadams/DARec/dataset/ratings_Toys_and_Games.csv", "/home2/dadams/DARec/dataset/ratings_Automotive.csv", train=True, preprocessed=True)
test_dataset = Mydata("/home2/dadams/DARec/dataset/ratings_Toys_and_Games.csv", "/home2/dadams/DARec/dataset/ratings_Automotive.csv", train=False, preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

print("Data is loaded")
# Get user/item dims
S_n_items, S_n_users = train_dataset.S_data.shape[0], train_dataset.S_data.shape[1]
T_n_items, T_n_users = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]

# Create two OT_AutoRec models: one for source, one for target
default_n_factors = args.n_factors
source_model = OT_AutoRec(n_users=S_n_users, n_items=S_n_items, n_factors=default_n_factors, ot_weight=args.ot_weight).cuda()
target_model = OT_AutoRec(n_users=T_n_users, n_items=T_n_items, n_factors=default_n_factors, ot_weight=args.ot_weight).cuda()

criterion = MRMSELoss().cuda()
optimizer = optim.Adam(list(source_model.parameters()) + list(target_model.parameters()), weight_decay=args.wd, lr=args.lr)

def train(epoch):
    source_model.train()
    target_model.train()
    Total_RMSE = 0
    Total_MASK = 0
    for idx, d in enumerate(train_loader):
        source_data = d[0].cuda()  # (batch, n_users)
        target_data = d[1].cuda()  # (batch, n_users)
        optimizer.zero_grad()
        # Forward: source and target, pass other domain for OT
        _, pred_s, ot_loss_s = source_model(source_data, x_other=target_data)
        _, pred_t, ot_loss_t = target_model(target_data, x_other=source_data)
        # RMSE loss
        loss_s, mask_s = criterion(pred_s, source_data)
        loss_t, mask_t = criterion(pred_t, target_data)
        # OT loss (average if both available)
        ot_losses = []
        if ot_loss_s is not None:
            ot_losses.append(ot_loss_s)
        if ot_loss_t is not None:
            ot_losses.append(ot_loss_t)
        ot_loss = sum(ot_losses) / len(ot_losses) if ot_losses else 0.0
        # Total loss
        total_loss = loss_s + loss_t + args.ot_weight * ot_loss
        Total_RMSE += loss_s.item() + loss_t.item()
        Total_MASK += torch.sum(mask_s).item() + torch.sum(mask_t).item()
        total_loss.backward()
        optimizer.step()
    return math.sqrt(Total_RMSE / Total_MASK)

def test():
    source_model.eval()
    target_model.eval()
    Total_RMSE = 0
    Total_MASK = 0
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            source_data = d[0].cuda()
            target_data = d[1].cuda()
            _, pred_s, _ = source_model(source_data)
            _, pred_t, _ = target_model(target_data)
            loss_s, mask_s = criterion(pred_s, source_data)
            loss_t, mask_t = criterion(pred_t, target_data)
            Total_RMSE += loss_s.item() + loss_t.item()
            Total_MASK += torch.sum(mask_s).item() + torch.sum(mask_t).item()
    return math.sqrt(Total_RMSE / Total_MASK)

if __name__=="__main__":
    train_rmse = []
    test_rmse = []
    wdir = r"./"
    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(test())
        if epoch % args.epochs == args.epochs - 1:
            torch.save(source_model.state_dict(), wdir + "S_OT_AutoRec_%d.pkl" % (epoch+1))
            torch.save(target_model.state_dict(), wdir + "T_OT_AutoRec_%d.pkl" % (epoch+1))
    print(min(test_rmse))
    plt.plot(range(args.epochs), train_rmse, range(args.epochs), test_rmse)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.xticks(range(0, args.epochs, 2))
    plt.show()