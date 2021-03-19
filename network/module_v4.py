import torch.nn as nn
class AlexNet(nn.Module): #1*120*120
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=11,stride=4,padding=2), #96*29*29
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2) #96*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size=5,padding=2), #256*14*14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2) #256*6*6
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3,padding=1), #384*6*6
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384,384,kernel_size=3,padding=1),#384*6*6
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,256,kernel_size=3,padding=1), #256*6*6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2) #256*2*2
        )

        self.fc1 = nn.Sequential(
            nn.Linear(13824,4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096,6)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out