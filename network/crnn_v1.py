import torch
import torch.nn as nn

#---------------CNN结构-----------------------------
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),  # 16*116
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*58
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),  # 32*56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*28
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 32*26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*13
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 64*11
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*5*5
        )

        self.drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128*5*5,1200)
        self.fc2 = nn.Linear(1200,512)
        self.fc3 = nn.Linear(512,300)

    def forward(self,x_3d):
        cnn_embed_seq = []
        # 分别对序列中的每一张图片做CNN
        for t in range(x_3d.size(2)):
            #CNNs
            out = self.conv1(x_3d[:, :, t, :, :])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = out.view(out.size(0),-1)
            #FCs
            out = nn.ReLU()(self.fc1(out))
            out = nn.ReLU()(self.fc2(out))
            out = self.drop(out)
            out = self.fc3(out)

            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq

#---------------RNN结构-----------------------------
class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()

        self.LSTM = nn.LSTM(
            input_size = 300,
            hidden_size = 256,
            num_layers = 3,
            batch_first=True
        )

        self.drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self,x_rnn):

        self.LSTM.flatten_parameters()
        RNN_out,(h_n,h_c) = self.LSTM(x_rnn,None)

        out = self.fc1(RNN_out[:, -1, :])
        out = nn.ReLU()(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out



