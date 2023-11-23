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




def CRFA_train(data_path, tmp_path, emotion, stream, batch_size=1024, num_epochs=200, 
               lr=1e-3, window_size = 9, model = "CRFA"):
    
    X, y = load_data(data_path, emotion, stream, mode="train")
    num_channels, num_points = X.shape[1], X.shape[2]

    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold_idx, (idx_train, idx_test) in enumerate(cv.split(X)):
        
        X_train, y_train, X_test, y_test = X[idx_train], y[idx_train], X[idx_test], y[idx_test]

        # save index of fold 
        np.savez(tmp_path+"fold_{}_".format(fold_idx) + "index.npz", index_train = idx_train, index_test = idx_test)
        
        train_loader = build_dataloader(X_train, y_train, batch_size, True)
        test_loader = build_dataloader(X_test, y_test, batch_size, True)
        
        net = CRFAEmotionNet(num_channels=num_channels, num_points=num_points, window_size=window_size, mode=model)
        
        optimizer = optim.Adam(net.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()
        
        
        since = time.time()
        net, history = train(net, 
                             train_loader=train_loader, 
                             test_loader=test_loader, 
                             num_epochs=num_epochs, 
                             optimizer=optimizer,
                             loss_function=loss_function,
                             is_early_stopping=True,
                             early_stop_step=10)
        
        print("time:", time.time() - since)
        print("param:", sum(param.numel() for param in net.parameters()))
        
        torch.save(net, tmp_path + "{}/fold_{}_net.pth".format(stream, fold_idx))
        
if __name__ == "__main__":
    path = '/home/asher/DEAP/deap/'
    tmp = 'tmp/'
    CRFA_train(path, tmp, 0, stream='dynamic', num_epochs=200)
    CRFA_train(path, tmp, 0, stream='static', num_epochs=200)
    # CRFA_train(path, tmp, 1, stream='dynamic', num_epochs=200)
    # CRFA_train(path, tmp, 1, stream='static', num_epochs=200)