from torch import nn
import torch.nn.functional as F
import torch
from .CRFABlock import CRFABlock

class CRFAEmotionNet(nn.Module):
    def __init__(self, num_channels, num_points, window_size=9, dropout=0.5, ratio=2, mode='CRFA'):
        super(CRFAEmotionNet, self).__init__()
        
        self.crfa = CRFABlock(num_channels, num_points, window_size, dropout, ratio, mode)
        
        num_features = num_points - window_size + 1 if not (mode == "NoCR" or mode == "NoCRFA") else num_points
        
        self.classify = nn.Sequential(
            nn.Linear(in_features = num_features * num_channels, out_features = 128),
            nn.BatchNorm1d(num_features= 128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.crfa(x)
        x = x.view(x.size(0), -1 )
        return self.classify(x)

if __name__ == "__main__":
    x = torch.randn(size = (10, 32, 128))
    net1 = CRFAEmotionNet(32, 128, mode="NoCR")
    net2 = CRFAEmotionNet(32, 128, mode="NoFA")
    net3 = CRFAEmotionNet(32, 128, mode="NoCRFA")
    net4 = CRFAEmotionNet(32, 128, mode="CRFA")
    print(net1(x).size())
    print(net2(x).size())
    print(net3(x).size())
    print(net4(x).size())