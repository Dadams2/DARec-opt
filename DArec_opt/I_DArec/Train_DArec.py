import numpy as np
import torch.optim as optim
import torch.utils.data
from I_DArec import *
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

def get_default_args():
    class Args:
        pass
    args = Args()
    args.epochs = 70
    args.batch_size = 64
    args.lr = 1e-3
    args.wd = 1e-4
    args.n_factors = 200
    args.n_users = 1637
    args.S_n_items = 23450
    args.T_n_items = 16993
    args.RPE_hidden_size = 200
    args.S_pretrained_weights = r'Pretrained_ParametersS_AutoRec_50.pkl'
    args.T_pretrained_weights = r'Pretrained_ParametersT_AutoRec_50.pkl'
    return args

def parse_args(args_dict=None):
    """
    If args_dict is None, parse from command line. Otherwise, use args_dict to override defaults.
    """
    import argparse
    parser = argparse.ArgumentParser(description='DArec with PyTorch')
    parser.add_argument('--epochs', '-e', type=int, default=70)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-4)
    parser.add_argument("--n_factors", type=int, default=200, help="embedding dim")
    parser.add_argument("--n_users", type=int, default=1637, help="size of each image batch")
    parser.add_argument("--S_n_items", type=int, default=23450, help="Source items number")
    parser.add_argument("--T_n_items", type=int, default=16993, help="Target items number")
    parser.add_argument("--RPE_hidden_size", type=int, default=200, help="hidden size of Rating Pattern Extractor")
    parser.add_argument("--S_pretrained_weights", type=str, default=r'Pretrained_ParametersS_AutoRec_50.pkl')
    parser.add_argument("--T_pretrained_weights", type=str, default=r'Pretrained_ParametersT_AutoRec_50.pkl')
    if args_dict is None:
        return parser.parse_args()
    else:
        args = parser.parse_args([])
        for k, v in args_dict.items():
            setattr(args, k, v)
        return args


def run_training(args=None):
    """
    Main training entry point. Accepts an args object (from parse_args or get_default_args), or None for defaults.
    Returns: train_rmse, test_rmse
    """
    if args is None:
        args = get_default_args()

    # Load Data
    train_dataset = Mydata("/home2/dadams/DARec/dataset/ratings_Toys_and_Games.csv", "/home2/dadams/DARec/dataset/ratings_Automotive.csv", train=True, preprocessed=True)
    test_dataset = Mydata("/home2/dadams/DARec/dataset/ratings_Toys_and_Games.csv", "/home2/dadams/DARec/dataset/ratings_Automotive.csv", train=False, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    args.n_users = train_dataset.S_data.shape[1]
    args.S_n_items, args.T_n_items = train_dataset.S_data.shape[0], train_dataset.T_data.shape[0]

    print("Data is loaded")
    # neural network
    net = I_DArec(args)
    net.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
    net.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
    net.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=args.wd, lr=args.lr)
    RMSE = MRMSELoss().cuda()
    criterion = DArec_Loss().cuda()

    def train(epoch):
        Total_RMSE = 0
        Total_MASK = 0
        for idx, d in enumerate(train_loader):
            source_rating, target_rating, source_labels, target_labels = d
            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels= target_labels.squeeze(1).long().cuda()

            optimizer.zero_grad()
            is_source = True
            if is_source == True:
                class_output, source_prediction, target_prediction = net(source_rating, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                        source_rating, target_rating, source_labels)
                rmse, _ = RMSE(source_prediction, source_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()
                loss = source_loss
            is_source = False
            if is_source == False:
                class_output, source_prediction, target_prediction = net(target_rating, is_source)
                target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                                  source_rating, target_rating, target_labels)
                loss += target_loss
            loss.backward()
            optimizer.step()
        return math.sqrt(Total_RMSE / Total_MASK)

    def test():
        Total_RMSE = 0
        Total_MASK = 0
        with torch.no_grad():
            for idx, d in enumerate(test_loader):
                source_rating, target_rating, source_labels, target_labels = d
                source_rating = source_rating.cuda()
                target_rating = target_rating.cuda()
                source_labels = source_labels.squeeze(1).long().cuda()
                target_labels = target_labels.squeeze(1).long().cuda()
                is_source = True
                if is_source == True:
                    class_output, source_prediction, target_prediction = net(source_rating, is_source)
                    source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                            source_rating, target_rating, source_labels)
                    rmse, _ = RMSE(source_prediction, source_rating)
                    Total_RMSE += rmse.item()
                    Total_MASK += torch.sum(target_mask).item()
                    loss = source_loss
        return math.sqrt(Total_RMSE / Total_MASK)

    train_rmse = []
    test_rmse = []
    wdir = r"I-darec"
    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(test())
        if epoch % args.epochs == args.epochs - 1:
            torch.save(net.state_dict(), wdir+"%d.pkl" % (epoch+1))
    print(min(test_rmse))
    # Improved plot with seaborn
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), train_rmse, label='Train RMSE', marker='o', linewidth=2)
    plt.plot(range(args.epochs), test_rmse, label='Test RMSE', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.title('Training and Test RMSE over Epochs', fontsize=16)
    plt.xticks(range(0, args.epochs, max(1, args.epochs // 20)))
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return train_rmse, test_rmse

def main(args_dict=None):
    """
    Main entry point for CLI or notebook. args_dict can be a dictionary of arguments to override defaults.
    """
    if args_dict is not None:
        args = parse_args(args_dict)
    else:
        import sys
        if hasattr(sys, 'argv') and len(sys.argv) > 1:
            args = parse_args()
        else:
            args = get_default_args()
    return run_training(args)

if __name__=="__main__":
    main()