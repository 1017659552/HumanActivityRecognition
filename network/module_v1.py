import torch.nn as nn

class Net(nn.Module):  # 1*120*120
    def __init__(self):
        super (Net,self).__init__()

        self.cnn_layers = nn.Sequential(
            #第一个2D卷积层
            nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1), #4*120*120
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), #4*60*60

            #第二个卷积层
            nn.Conv2d(4,8,kernel_size=3,stride=1,padding=1), #8*60*60
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # 8*30*30

            # 第二个卷积层
            # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 16*30*30
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)  # 16*15*15

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(8 * 30 * 30, 120),
            nn.Linear(120,6)##########################这里记得修改类别数
        )
    #前项传播
    def forward(self,x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0),-1)
        x = self.linear_layers(x)
        return x
