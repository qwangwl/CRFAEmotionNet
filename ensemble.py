import os
import numpy as np
import torch
from torch import nn, optim
from utils.utils import train, test
from utils.dataset import load_data, build_dataloader
from sklearn.model_selection import KFold
import time
from model.Network import CRFAEmotionNet

from torch.utils.data import TensorDataset, DataLoader


SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_for_stream(X_test, tmp_path, stream, fold_idx=0):
    
    if stream == "dynamic":
        X_test = np.diff(X_test, n=2, axis=-1)
        
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    X_test = X_test.to(device)
    
    net_path = tmp_path + "{}/fold_{}_net.pth".format(stream, fold_idx)
    net = torch.load(net_path)
    
    return net(X_test).cpu()



def compute_acc(y_pred, y_true):
    y_pred = torch.max(y_pred.data, dim=1)[1]
    TP = ((y_true == 1) & (y_pred.data == 1)).sum()
    TN = ((y_true == 0) & (y_pred.data == 0)).sum()
    FP = ((y_true == 1) & (y_pred.data == 0)).sum()
    FN = ((y_true == 0) & (y_pred.data == 1)).sum()
    
    acc = (TP+TN) / (TP+TN+FP+FN)
    P = TP / (TP+FP)
    R = TP / (TP+FN)
    F1 = 2 * P * R / (R+P)
    return acc, F1

def CRFA_ensemble(tmp_path, emotion):
    X, y = load_data(None, emotion, stream="static", mode="test")
    
    static_acc = []
    static_f1 = []
    dynamic_acc = []
    dynamic_f1 = []
    fusion_acc = []
    fusion_f1 = []
    
    for i in range(10):
        idx_test = np.load(tmp_path+"fold_{}_index.npz".format(i))["index_test"]
        
        X_test, y_test =X[idx_test], y[idx_test]
        
        y_test = torch.from_numpy(y_test).type(torch.long)
        static_y_pred = predict_for_stream(X_test, tmp_path, "static", i)
        dynamic_y_pred = predict_for_stream(X_test, tmp_path, "dynamic", i)
        
        fusion_y_pred  = static_y_pred + dynamic_y_pred
        
        static_acc_, static_f1_ = compute_acc(static_y_pred, y_test)
        dynamic_acc_, dynamic_f1_ = compute_acc(dynamic_y_pred, y_test)
        fusion_acc_, fusion_f1_ = compute_acc(fusion_y_pred, y_test)
        
        static_acc.append(static_acc_)
        static_f1.append(static_f1_)
        dynamic_acc.append(dynamic_acc_)
        dynamic_f1.append(dynamic_f1_)
        fusion_acc.append(fusion_acc_)
        fusion_f1.append(fusion_f1_)
        
        
    print('static:{:.3f}+{:.3f}'.format(np.mean(static_acc), np.std(static_acc)))
    print('dynamic:{:.3f}+{:.3f}'.format(np.mean(dynamic_acc), np.std(dynamic_acc)))
    print('fusion:{:.3f}+{:.3f}'.format(np.mean(fusion_acc), np.std(fusion_acc)))
    
if __name__ == "__main__":
    CRFA_ensemble("tmp/CRFA/valence", 0)
    # CRFA_ensemble("tmp/CRFA/arousal", 1)