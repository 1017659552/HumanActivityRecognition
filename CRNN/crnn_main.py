import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.preprocess import MyDataset
from  network import crnn_v1 as CRNN
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix

#--------------------定义参数-----------------------------------
# root_dir = 'D:\\SWUFEthesis\\data\\KTH'
# process_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess_v3'
Dataset_labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
crop_size = 120
frame_width = 160
frame_height = 120

clip_len = 16 #帧序列的长度
n_epochs = 250
n_batch_size = 32
n_lr = 1e-5

def plot_confusion_matrix(C):
    plt.matshow(C, cmap=plt.cm.Greens)
    plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#--------------------------------------------------------------

print("正在读取数据集... ...")
data_train = DataLoader(MyDataset(split='train', clip_len=clip_len,get3d_data = True),batch_size=n_batch_size, shuffle=True)
data_test = DataLoader(MyDataset(split='test', clip_len=clip_len,get3d_data = True), batch_size=n_batch_size, shuffle=True)
data_val = DataLoader(MyDataset(split='val', clip_len=clip_len,get3d_data = True), batch_size=n_batch_size, shuffle=True)

# 检查gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("use device:",device)

module_CNN = CRNN.EncoderCNN()
module_RNN = CRNN.DecoderRNN()
module_CNN.to(device)
module_RNN.to(device)

crnn_params = list(module_CNN.parameters()) + list(module_RNN.parameters())
optimizer = Adam(crnn_params,lr = n_lr)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_val_losses = []
epoch_val_scores = []

for epoch in tqdm(range(n_epochs)):
    #-----------训练模块-------------------------------------------
    train_losses = []
    train_scores = []
    module_CNN.train()
    module_RNN.train()
    for inputs, labels in data_train:
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device, dtype=torch.int64)

        optimizer.zero_grad() #清除梯度

        output_CNN = module_CNN(inputs)
        output_RNN = module_RNN(output_CNN) #[batch_size，类别数]

        probs = nn.Softmax(dim=1)(output_RNN)
        loss = F.cross_entropy(probs,labels)
        train_losses.append(loss.item())

        y_pred = torch.max(probs, 1)[1]  # y_pred != output
        step_score = accuracy_score(labels.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        train_scores.append(step_score)

        loss.backward()
        optimizer.step()
        # print("epoch "+str(epoch)+" : loss: {},score:{}")

    epoch_train_losses.append(np.mean(train_losses))
    epoch_train_scores.append(np.mean(train_scores))
    # -----------结束-测试模块-------------------------------------------

    # -----------验证模块-------------------------------------------
    val_losses = []
    val_scores = []
    module_CNN.eval()
    module_RNN.eval()
    for inputs, labels in data_val:
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device, dtype=torch.int64)

        optimizer.zero_grad()

        output_CNN = module_CNN(inputs)
        output_RNN = module_RNN(output_CNN)  # [batch_size，类别数]

        probs = nn.Softmax(dim=1)(output_RNN)
        loss = F.cross_entropy(probs, labels)
        val_losses.append(loss.item())

        y_pred = torch.max(probs, 1)[1]  # y_pred != output
        step_score = accuracy_score(labels.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        val_scores.append(step_score)

        # print(output_RNN)
    epoch_val_losses.append(np.mean(val_losses))
    epoch_val_scores.append(np.mean(val_scores))
    # -----------结束-测试模块-------------------------------------------

    print("Epoch:{}, Training Loss:{}, Valid Loss:{}".format(epoch, np.mean(train_losses), np.mean(val_losses)))
    print("Epoch:{}, Training Score:{}, Valid Score:{}".format(epoch, np.mean(train_scores), np.mean(val_scores)))
    # print("Epoch:{}, Training Loss:{}".format(epoch, np.mean(train_losses)))

plt.plot(epoch_train_losses,label = 'Training losses')
plt.plot(epoch_val_losses,label = 'Validation losses')
plt.legend()
plt.show()

plt.plot(epoch_train_scores,label = 'Training scores')
plt.plot(epoch_val_scores,label = 'Validation scores')
plt.legend()
plt.show()

#-----------测试模块-------------------------------------------
test_scores = []
for inputs, labels in data_test:
    inputs = Variable(inputs, requires_grad=True).to(device)
    labels = Variable(labels).to(device, dtype=torch.int64)

    output_CNN = module_CNN(inputs)
    output_RNN = module_RNN(output_CNN)  # [batch_size，类别数]

    probs = nn.Softmax(dim=1)(output_RNN)
    loss = F.cross_entropy(probs, labels)

    y_pred = torch.max(probs, 1)[1]  # y_pred != output
    step_score = accuracy_score(labels.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
    test_scores.append(step_score)

    c = confusion_matrix(labels.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
    plot_confusion_matrix(c)

print("Test Accuracy : {} %".format(np.mean(test_scores)))
#-----------结束-测试模块-------------------------------------------