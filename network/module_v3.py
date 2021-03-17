import  torch.nn as nn
class Net2(nn.Module): # 1*120*120
    # def __init__(self):
    #     super(Net2, self).__init__()
    #
    #     self.conv1 = nn.Sequential(
    #         nn.Conv2d(1,16,kernel_size = 5), # 16*116*116
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2), # 16*58*58
    #         # nn.Dropout(0.5)
    #     )
    #
    #     self.conv2 = nn.Sequential(
    #         nn.Conv2d(16,32,kernel_size=3), # 32*56*56
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2), #32*28*28
    #         # nn.Dropout(0.5)
    #     )
    #
    #     self.conv3 = nn.Sequential(
    #         nn.Conv2d(32,64,kernel_size=3), # 64*26*26
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2), # 64*13*13
    #         # nn.Dropout(0.5)
    #     )
    #
    #     self.fc1 = nn.Linear(10816,128)
    #     self.dropfc1 = nn.Dropout(0.5)
    #     self.fc2 = nn.Linear(128,6)

    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = 3,padding=1), # 16*120*120
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16*60*60
            # nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=1), # 32*60*60
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #32*30*30
            # nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3), # 32*28*28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #32*14*14
            # nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3), # 64*12*12
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64*6*6
            # nn.Dropout(0.5)
        )


        self.fc1 = nn.Linear(128*6*6,4096)
        self.dropfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096,6)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = out.view(out.size(0),-1)
        # print(out.shape)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropfc1(out)
        out = self.fc2(out)

        return out