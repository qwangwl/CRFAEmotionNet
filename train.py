import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import KFold
import time
import pickle
import argparse

from trainer import train, test
from dataset import load_data, build_dataloader,normalization
from Models import CRFAEmotionNet
SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def CRFA_train(X, y, tmp_path, batch_size=1024, num_epochs=200, 
               lr=1e-3, window_size = 9, nb_class=2, dataset="deap"):
    
    num_channels, num_points = X.shape[1], X.shape[2]
    acc = []
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold_idx, (idx_train, idx_test) in enumerate(cv.split(X)):
        
        X_train, y_train, X_test, y_test = X[idx_train], y[idx_train], X[idx_test], y[idx_test]
        if dataset == "seed":
            X_train = normalization(X_train)
            X_test = normalization(X_test)
        train_loader = build_dataloader(X_train, y_train, batch_size, True)
        test_loader = build_dataloader(X_test, y_test, batch_size, True)
        
        net = CRFAEmotionNet(num_channels=num_channels, num_points=num_points, window_size=window_size, nb_class=nb_class)
        
        optimizer = optim.Adam(net.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()
        
        since = time.time()
        net, history = train(net, 
                             train_loader=train_loader, 
                             test_loader=test_loader, 
                             num_epochs=num_epochs, 
                             optimizer=optimizer,
                             criterion=loss_function,
                             is_early_stopping=True,
                             early_stop_step=10)
        
        acc.append(history["test_acc"][-1])
        print("fold: {}, param: {},  time: {}".format(fold_idx, sum(param.numel() for param in net.parameters()), time.time() - since))
        
        # torch.save(net, tmp_path + "/fold_{}_net.pth".format(fold_idx))
        with open("{}\\fold_{}.pkl".format(tmp_path, fold_idx), "wb") as f:
            pickle.dump(history, f)
    print(np.mean(acc), np.std(acc))
    return acc


# for deap
def train_subject_independent(args):
    if args.emotion == "arousal":
        X, y, _ = load_data(args.path, emotion=1)
    elif args.emotion == "valence":
        X, y, _ = load_data(args.path, emotion=0)
        
    CRFA_train(X, y, args.tmp+"{}\\".format(args.emotion),  num_epochs=200)

# for deap
def train_subject_dependent(args):
    if args.emotion == "arousal":
        X, y, Group = load_data(args.path, emotion=1)
    elif args.emotion == "valence":
        X, y, Group = load_data(args.path, emotion=0)
    all_acc = []
    for sub in range(1, 33):
        sub_X = X[Group == sub]
        sub_y = y[Group == sub] 
        acc = CRFA_train(sub_X, sub_y, args.tmp + "{}\\".format(sub)+"{}\\".format(args.emotion),  num_epochs=200) 
        all_acc.append(acc)
    np.savez("tmp\\{}.npz".format(args.emotion), np.array(all_acc))

# for seed
def train_subject_independent_for_seed(args):
    X, y, _ = load_data(args.path, data_name="SEED", session=args.session)  
    CRFA_train(X, y, args.tmp+"seed\\{}\\".format(args.session),  num_epochs=200, window_size=1, nb_class=3, dataset="seed")



    

if __name__ == "__main__":
    path = "EEG_DataSets\\DEAP\\"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--emotion', type=str, default="arousal")
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--tmp', type=str, default="tmp\\")
    parser.add_argument('--session', type=int, default=2)
    args = parser.parse_args()
    
    # for deap SID
    train_subject_independent(args)
    # for deap SD
    # train_subject_dependent(args)
    # for SEED SD
    # train_subject_independent_for_seed(args)
    
    