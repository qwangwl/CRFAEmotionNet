import os
import numpy as np
import torch
from torch import nn, optim
import pickle
from torch.utils.data import TensorDataset, DataLoader


def load_deap(path):
    # extract data from dat file
    with open(path, 'rb') as f:
        subject = pickle.load(f, encoding = 'latin1')    

    data = subject['data']
    target = subject['labels']
    
    return data[:, np.arange(32)], target[:, np.arange(2)] # get 32 eeg channels and valence/arousal 


def intercept_signal(signal, start, stop=None, sf=128):
    # intercept a segment of the signal
    if stop is None:
        start, stop = 0, start

    point_of_start, point_of_end = start*sf, stop*sf

    return signal[..., point_of_start:point_of_end]


def split_signal(signal, labels,  window=1, step=None, sf=128):
    # Slice the signal according to the specified window and step size
    if step is None:
        step = window
    
    data = []

    start, stop, step = 0, int(window*sf), int(step*sf)
    
    # slice signal
    while stop <= signal.shape[-1] or start == 0:
        data.append(signal[..., start:stop])
        start += step
        stop += step
    
    # Re-stack the sliced data
    data  = np.stack(data, axis=-3)
    trials, slices, *eeg_shape = data.shape
    data = data.reshape(-1, *eeg_shape)
    
    # Assign a group to a slice for further processing
    groups = np.arange(1, trials+1)
    groups = np.repeat(groups, slices)
    
    # If labeled, each slice has the same label as the video
    if labels is not None:
        labels = np.repeat(labels, slices, axis=0)
    
    return data, labels, groups
    
def stimu_minus_base(dbase, gbase, dstimu, gstimu):
    # If the base group is the same as the stimu group, then subtract
    ChangeEEG = np.zeros_like(dstimu)
    for g in np.unique(gbase):
        # 每一组进行不同的处理
        BaseMean = np.mean(dbase[gbase==g], axis=0)
        ChangeEEG[gstimu==g] = dstimu[gstimu==g] - BaseMean
    return ChangeEEG


def label_binarizer(label, threshold=5):
    tmp = np.zeros_like(label, dtype=np.int8)
    tmp[label< threshold] = 0
    tmp[label>=threshold] = 1
    return tmp

def load_data(data_path, emotion, stream="static", mode="train"):
    if mode == "train" and data_path is not None and not os.path.exists("tmp/data.npz"):
        subjects = ["s{:02d}.dat".format(i) for i in range(1, 33)]
        Data = []
        Label = []
        
        for sub_idx, sub in enumerate(subjects):
            # first load deap
            sub_data, sub_label = load_deap(data_path + sub)
            
            # second intercept signal
            sub_base = intercept_signal(sub_data, start=0, stop=3)  # base signal
            sub_stimu = intercept_signal(sub_data, start=3, stop=63) # stimu signal
            
            # third Split the signal, turning it into 1s segments
            sub_base, _, sub_base_g = split_signal(sub_base, None)
            sub_stimu, sub_label, sub_stimu_g = split_signal(sub_stimu, sub_label)
            
            # in the end, remove base
            result = stimu_minus_base(sub_base, sub_base_g, sub_stimu, sub_stimu_g)
        
            Data.append(result)
            Label.append(sub_label)
        
        Data = np.concatenate(Data)
        Label = np.concatenate(Label)
        
        np.savez("tmp/data.npz", Data=Data, Label=Label)
    else:
        eeg = np.load("tmp/data.npz")
        Data, Label = eeg["Data"], eeg["Label"]
    
    # for dynamic representation
    if stream == "dynamic":
        Data = np.diff(Data, n=2, axis=-1)
    
    return Data, label_binarizer(Label[:, emotion])


def build_dataloader(X, y, batch_size=512, shuffle=True, num_workers=8):
    
    X_torch = torch.from_numpy(X).type(torch.Tensor)
    y_torch = torch.from_numpy(y).type(torch.long)
    
    dataset = TensorDataset(X_torch, y_torch)
    
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size, 
        shuffle = shuffle,
        num_workers = num_workers     
    )
    return data_loader

if __name__ == "__main__":
    x = np.random.random_sample(size = (40, 32, 8064)) 
    y = np.random.random_sample(size = (40, 2))
    print(x.shape)
    d, l = split_signal(x, y)
    print(d.shape, l.shape)
        