#LeNet网络
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module): # 1*120*120
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*27*27, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 6)
    def forward(self, x):
        out = F.relu(self.conv1(x)) #6*116*116
        out = F.max_pool2d(out, 2) #6*58*58
        out = F.relu(self.conv2(out)) #16*54*54
        out = F.max_pool2d(out, 2) #16*27*27
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out