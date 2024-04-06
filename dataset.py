import os
import numpy as np
import torch
from torch import nn, optim
import pickle
from torch.utils.data import TensorDataset, DataLoader


def load_pkl(path):
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

def load_deap(data_path, emotion):
    if data_path is not None and not os.path.exists("tmp/data.npz"):
        subjects = ["s{:02d}.dat".format(i) for i in range(1, 33)]
        subjects = list(subjects)
        Data = []
        Label = []
        Group  = []
        for sub_idx, sub in enumerate(subjects):
            # first load deap
            sub_data, sub_label = load_pkl(data_path + sub)
            
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
            Group.append((sub_idx+1) * np.ones(shape=len(sub_label), dtype=np.int8))
        Data = np.concatenate(Data)
        Label = np.concatenate(Label)
        Group = np.concatenate(Group)
        np.savez("tmp/data.npz", Data=Data, Label=Label, Group =Group)
    else:
        eeg = np.load("tmp/data.npz")
        Data, Label, Group = eeg["Data"], eeg["Label"], eeg["Group"]
        
    return Data, label_binarizer(Label[:, emotion]), Group


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

import scipy.io as scio

def load_seed_frommat_with_de(eeg_path, label_path):
    # 从一个文件中获取数据，主要是提取DE特征
    eeg = scio.loadmat(eeg_path)
    label = scio.loadmat(label_path)["label"][0]
    
    Group = None
    Data = None
    Label = None
    
    # 获取DE的key值
    de_keys = [key for key in eeg.keys() if key.startswith('de_LDS')]
    
    for i, key in enumerate(de_keys):
        
        # 第i个影片的数据
        eeg_trial = eeg[key].transpose(1, 0, 2)
        
        # print(eeg_trial.shape, label_trial)
        # 对当前影片创建分组，当前每一个切片的影片都属于当前影片
        group_trial = np.ones((eeg_trial.shape[0], 1), dtype=np.int16) * (i+1)
        # 对标签进行填充
        label_trial = np.full((eeg_trial.shape[0]), label[i])
        
        # print(group_trial.shape, eeg_trial.shape, label_trial.shape)
        # 对数据进行拼接
        if Data is not None:
            # print(Group.shape, group_trial.shape)
            Group = np.concatenate((Group, group_trial))
            Data = np.concatenate((Data, eeg_trial))
            Label = np.concatenate((Label, label_trial))
        else:
            Group = group_trial
            Data = eeg_trial
            Label = label_trial
        
    return Group, Data, Label
    # return Data, Label

def load_seed(data_path, session=1):
    # 获取某一个session中的所有受试者数据
    subs = os.listdir(data_path + str(session))
    subs.sort() 
    
    Group = None
    Data = None
    Label = None
    
    for sub in subs:
        trial_G, D, L = load_seed_frommat_with_de(data_path+str(session) + "/" + sub, data_path + "label.mat")
        # print(trial_G.shape, D.shape, L.shape)
        # 对当前受试者进行分组
        sub_G = np.ones((D.shape[0], 1), dtype=np.int16) * int(sub.split("_")[0])
        
        # 拼合
        G = np.hstack((sub_G, trial_G))
        # print(G.shape, D.shape, L.shape)
        
        if Data is not None:
            Group = np.concatenate((Group, G))
            Data = np.concatenate((Data, D))
            Label = np.concatenate((Label, L))
        else:
            Group = G
            Data = D
            Label = L
        
    return Group, Data, Label

def load_data(data_path, data_name="DEAP",  emotion=1, session = 1):
    if data_name == "DEAP":
        return load_deap(data_path, emotion)
    elif data_name == "SEED":
        Group, Data, Label =  load_seed(data_path, session)
        Label += 1
        # Data = Data.reshape(-1, 310)
        Group = Group[:, 0] # 只需要受试者的分组
        return Data, Label, Group


def normalization(data):
    """
    Feature-wise normal
    """
    range_ = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.min(data, axis=0)) /  range_

if __name__ == "__main__":
    x = np.random.random_sample(size = (40, 32, 8064)) 
    y = np.random.random_sample(size = (40, 2))
    print(x.shape)
    d, l = split_signal(x, y)
    print(d.shape, l.shape)
        