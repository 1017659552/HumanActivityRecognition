import  torch.nn as nn
class Net2(nn.Module): # 1*120*120
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = 5), # 16*116
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 16*58
            # nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3), # 32*56
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), #32*28
            # nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3), # 32*26
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), #32*13
            # nn.Dropout(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3), # 64*11
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 64*5*5
            # nn.Dropout(0.5)
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128,128,kernel_size=3,padding=1), # 128*4*4
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,stride=2), # 128*2*2
        #     # nn.Dropout(0.5)
        # )
        #
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128*4*4
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 128*2*2
        #     # nn.Dropout(0.5)
        # )


        self.fc1 = nn.Linear(128*5*5,1200)
        # self.dropfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1200,6)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)


        out = out.view(out.size(0),-1)
        # print(out.shape)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        # out = self.dropfc1(out)
        out = self.fc2(out)

        return out