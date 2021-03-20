import torch
import torch.nn as nn

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride = 1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #
        # self.conv4a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))


        #三个全连接层
        # self.fc6 = nn.Linear(128*2*7*7, 4096)
        self.fc6 = nn.Linear(64*4*15*15, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 6)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        #
        # x = self.relu(self.conv4a(x))
        # x = self.relu(self.conv4b(x))
        # x = self.pool4(x)
        # print(x.shape)


        x = x.view(x.size(0),-1)
        x = self.relu(self.fc6(x))
        # print(x.shape)
        x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)
        # print(logits.shape)
        return logits
