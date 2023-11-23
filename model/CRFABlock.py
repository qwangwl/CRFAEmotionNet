from torch import nn
import torch.nn.functional as F
import torch

class CRBlock(nn.Module):
    def __init__(self, num_channels, window_size=9, dropout=0.5):
        super(CRBlock, self).__init__()
        self.cr = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=window_size)
        self.bn = nn.BatchNorm1d(num_features=num_channels)
        self.dropout = nn.Dropout1d(dropout)
    
    def forward(self, x):
        # x shape (B, C, T)
        f = self.cr(x)
        f = self.bn(f)
        f = F.relu(f)
        f = self.dropout(f)
        return f
    

class FABlock(nn.Module):
    def __init__(self, num_features, ratio=2):
        super(FABlock, self).__init__()
        
        self.cGAP = nn.AdaptiveAvgPool1d(1)
        self.f_linear = nn.Conv1d(in_channels=num_features, out_channels=num_features//ratio, kernel_size=1)
        self.s_linear = nn.Conv1d(in_channels=num_features //ratio, out_channels=num_features, kernel_size=1)
        
    def forward(self, x):
        # x shape (B, N, C)
        f = self.cGAP(x)
        
        f = self.f_linear(f)
        f = F.relu(f)
        f = self.s_linear(f)
        
        return torch.sigmoid(f) * x
 
class CRFABlock(nn.Module):
    def __init__(self, num_channels, num_points, window_size=9, dropout=0.5, ratio=2, mode='CRFA'):
        super(CRFABlock, self).__init__()
        self.mode = mode
        if not (self.mode == "NoCR" or self.mode == "NoCRFA"):
            self.crblock = CRBlock(num_channels, window_size, dropout)
        
        num_features = num_points - window_size + 1 if not (self.mode == "NoCR" or self.mode == "NoCRFA") else num_points
        
        if not (self.mode == "NoFA" or self.mode == "NoCRFA"):
            self.fablock = FABlock(num_features, ratio=ratio)
        
    
    def forward(self, x):
        # x shape (B, C, T)
        
        if not (self.mode == "NoCR" or self.mode == "NoCRFA"):
            x = self.crblock(x)
        
        if not (self.mode == "NoFA" or self.mode == "NoCRFA"):
            x = x.permute(0, 2, 1) # For the convenience of the code, transpose it
            x = self.fablock(x)
            x = x.permute(0, 2, 1) # reduction
        
        return x
        

if __name__ == "__main__":
    x = torch.randn(size=(10, 32, 126))
    net = CRBlock(num_channels=32)
    print(net(x).size())
    
    x = torch.randn(size = (10, 126, 32))
    net = FABlock(num_features=126)
    print(net(x).size())
    
    x = torch.randn(size = (10, 32, 128))
    net1 = CRFABlock(32, 128, mode="NoCR")
    net2 = CRFABlock(32, 128, mode="NoFA")
    net3 = CRFABlock(32, 128, mode="NoCRFA")
    net4 = CRFABlock(32, 128, mode="CRFA")
    print(net1(x).size())
    print(net2(x).size())
    print(net3(x).size())
    print(net4(x).size())