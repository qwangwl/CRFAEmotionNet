import numpy as np
import torch
from torch import nn, optim
from earlystopping import EarlyStopping

def train(net, 
          train_loader, 
          test_loader, 
          num_epochs=50, 
          optimizer=None, 
          criterion=None, 
          device='cuda:0', 
          is_early_stopping=True,
          early_stop_step = 5):
    
    if is_early_stopping:
        earlystoping = EarlyStopping(early_stop_step)
    
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    net = net.to(device)

    acc_train = []
    acc_test = []

    for epoch in range(1, num_epochs+1):
        
        net.train()
        for x_src, y_src in train_loader:
            
            x_src, y_src = x_src.to(device), y_src.to(device)
            
            # 同时更新
            # y_hat = net(x_src)
            # loss = criterion(y_hat, y_src)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # 先更新static
            s_y_hat = net(x_src, "static")
            loss = criterion(s_y_hat, y_src)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 再更新dynamic
            d_y_hat = net(x_src, "dynamic")
            loss = criterion(d_y_hat, y_src)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test
        acc1 = test(net, train_loader)
        acc2 = test(net, test_loader)
        
        acc_train.append(acc1)
        acc_test.append(acc2)
        
        # early stopping
        if is_early_stopping:
            earlystoping(acc1)
            #print(acc1, earlystoping.best_score, earlystoping.counter)
            if earlystoping.early_stop:
                print("Early stoping with epoch: %d and best train acc: %.5f and test acc: %.5f" % (epoch, acc1, acc2) )
                break
        if epoch % 20 == 0 or epoch == 1:
            print('epoch: %d, train acc: %.5f, test acc: %.5f' % (epoch, acc1, acc2))
        
    history = dict(
        train_acc = np.array(acc_train),
        test_acc = np.array(acc_test)
    )
    return net, history



def test(net, dataloader, device='cuda:0'):
    net.eval()
    n_total = 0
    n_correct = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            bs = len(y)
            
            y_hat = net.predict(x)
            y_pred = torch.max(y_hat, dim=1)[1]
            n_correct += y_pred.eq(y.data).sum().item()
            
            n_total += bs
    
    return n_correct / n_total


