import numpy as np
import torch
from torch import nn, optim
from .earlystopping import EarlyStopping

def train(net, 
          train_loader, 
          test_loader, 
          num_epochs=50, 
          optimizer=None, 
          loss_function=None, 
          device='cuda:0', 
          is_early_stopping=True,
          early_stop_step = 5):
    
    if is_early_stopping:
        earlystoping = EarlyStopping(early_stop_step)
    
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
        
    loss_function = loss_function.to(device)
        
    net = net.to(device)

    a_train = []
    a_test = []
    
    for epoch in range(num_epochs):
        
        net.train()
        # train
        for x_src, y_src in train_loader:
            
            net.zero_grad()
            
            x_src, y_src = x_src.to(device), y_src.to(device)
            
            y_hat = net(x_src)
            
            loss = loss_function(y_hat, y_src)
            
            loss.backward()
            optimizer.step()
        
        # test
        acc1 = test(net, train_loader)
        acc2 = test(net, test_loader)
        
        a_train.append(acc1)
        a_test.append(acc2)
        
        
        # early stopping
        if is_early_stopping:
            earlystoping(acc1)
            #print(acc1, earlystoping.best_score, earlystoping.counter)
            if earlystoping.early_stop:
                print("Early stoping with best train acc:", acc1, "and test acc:", acc2)
                break
        if epoch % 20 == 0:
            print('epoch: %d, train acc: %.4f, test acc: %.4f' % (epoch+1, acc1, acc2))
        
    history = dict(
        train_acc = np.array(a_train),
        test_acc = np.array(a_test)
    )
    print('epoch: %d, train acc: %.4f, test acc: %.4f' % (epoch+1, acc1, acc2))
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
            
            y_hat = net(x)
            y_pred = torch.max(y_hat, dim=1)[1]
            n_correct += y_pred.eq(y.data).sum().item()
            
            n_total += bs
    
    return n_correct / n_total


