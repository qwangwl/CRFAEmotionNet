from torch import nn
import torch.nn.functional as F
import torch

class CRBlock(nn.Module):
    def __init__(self, num_channels, window_size=9, dropout=0.5):
        super(CRBlock, self).__init__()
        self.cr = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=window_size, padding=window_size//2)
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
    def __init__(self, num_channels, num_points, window_size=9, dropout=0.5, ratio=2):
        super(CRFABlock, self).__init__()
        self.crblock = CRBlock(num_channels, window_size, dropout)
        self.fablock = FABlock(num_points, ratio=ratio)
        
    def forward(self, x):
        # x shape (B, C, T)
        
        x = self.crblock(x)
        x = x.permute(0, 2, 1) # For the convenience of the code, transpose it
        x = self.fablock(x)
        x = x.permute(0, 2, 1) # reduction`
        
        return x
        

class CRFAEmotionNet(nn.Module):
    def __init__(self, num_channels, num_points, window_size=9, dropout=0.5, nb_class=2, ratio=2):
        super(CRFAEmotionNet, self).__init__()
        
        self.static_fe = CRFABlock(num_channels, num_points, window_size, dropout, ratio)
        self.static_cls = nn.Sequential(
            nn.Linear(in_features = num_points * num_channels, out_features = 128),
            nn.BatchNorm1d(num_features= 128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=nb_class),
            nn.LogSoftmax(dim=1)
        )
        
        self.dynamic_fe = CRFABlock(num_channels, num_points-2, window_size, dropout, ratio)
        self.dynamic_cls = nn.Sequential(
            nn.Linear(in_features = (num_points-2) * num_channels, out_features = 128),
            nn.BatchNorm1d(num_features= 128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=nb_class),
            nn.LogSoftmax(dim=1) 
        )
        
    def forward(self, x,  mode = "static"):
        batch = x.size(0)
        if mode == "static":
            s_f = self.static_fe(x)
            s_f = s_f.view(batch, -1)
            s_y = self.static_cls(s_f)
            return s_y
        elif mode == "dynamic":
            dynamic_rep = torch.diff(x, n=2)
            d_f = self.dynamic_fe(dynamic_rep)
            d_f = d_f.view(batch, -1)
            d_y = self.dynamic_cls(d_f)
            return d_y
        
    def forward2(self, x):
        batch = x.size(0)
        s_f = self.static_fe(x)
        s_f = s_f.view(batch, -1)
        s_y = self.static_cls(s_f)
        dynamic_rep = torch.diff(x, n=2)
        d_f = self.dynamic_fe(dynamic_rep)
        d_f = d_f.view(batch, -1)
        d_y = self.dynamic_cls(d_f)
        return d_y + s_y
    
    
    def predict(self, x):
        batch = x.size(0)
        dynamic_rep = torch.diff(x, n=2)
        s_f = self.static_fe(x)
        s_f = s_f.view(batch, -1)
        s_y = self.static_cls(s_f)
        
        d_f = self.dynamic_fe(dynamic_rep)
        d_f = d_f.view(batch, -1)
        d_y = self.dynamic_cls(d_f)
        
        return s_y + d_y
    
    def static_predict(self, x):
        batch = x.size(0)
        s_f = self.static_fe(x)
        s_f = s_f.view(batch, -1)
        s_y = self.static_cls(s_f)
        
        return s_y

if __name__ == "__main__":
    x = torch.randn(size = (10, 32, 128))
    print(torch.diff(x, n=2).size())
    net4 = CRFAEmotionNet(32, 128)
    print(net4(x)[0].size())
    print(net4(x)[1].size())