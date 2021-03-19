from tqdm import tqdm
import  numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
import torch
from torch.optim import Adam,SGD
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data

import network.module_v1 as module_v1
import network.module_v2 as module_v2
import network.module_v3 as module_v3
import network.module_v4 as module_v4

########### 参数设置 #################
# root_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess_v2'
root_dir = '/home/mist/KTH_preprocess_v2'
labels = ['boxing','handclapping','handwaving','jogging','running','walking']
n_epochs = 30
n_batch_size = 32
n_lr = 1e-2

img_width = 120
img_height = 120
crop_size = 120
####################################
print(root_dir)

# 递归获取目录下所有文件名
def getPathImg(path,files):
    file_list = os.listdir(path)
    for file in file_list:
        path_cur = os.path.join(path,file)
        if os.path.isdir(path_cur):
            getPathImg(path_cur,files)
        else:
            files.append(os.path.join(path_cur))
    return files

# 获取图像的灰度矩阵和标签
def getImg(img_list):
    imgs_content = []
    imgs_label = []
    for img_name in tqdm(img_list):
        # 加载图片
        # img_name = "D:\\SWUFEthesis\\data\\KTH_preprocess\\train\\Jogging\\person01_jogging_d2_uncomp\\00044.jpg" #00044
        img = np.array(cv2.imread(img_name, cv2.IMREAD_GRAYSCALE))  # 读取灰度图像
        # 查看灰度直方图
        # print(img_name.split('\\')[5])
        # if(img_name.split('\\')[5]=='Jogging'):
        #     hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        #     # print(hist)
        #     plt.hist(img.ravel(), 256, [0,256])
        #     plt.show()
        #     # print(hist)
        # # 裁剪图片
        y0 = int((img_height - crop_size) / 2)
        x0 = int((img_width - crop_size) / 2)
        img = img[y0:crop_size + y0, x0:crop_size + x0]

        img = img / 255.0  # 归一化像素值
        img = img.astype(np.float32)
        imgs_content.append(img)
        # 加载标签
        # label = img_name.split('\\')[-2].split('_')[1] ######################本地
        label = img_name.split('/')[-2].split('_')[1]  #######################远程
        imgs_label.append(labels.index(label))

    assert len(imgs_label) == len(imgs_content), '图像和标签数量不一致，退出程序！'
    imgs_content = np.array(imgs_content)
    imgs_label = np.array(imgs_label)
    return  imgs_content,imgs_label

def to_tensor(x,y):
    # 将训练集转换成torch张量
    x_len = len(x)
    x = x.reshape(x_len, 1, crop_size, crop_size)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    # print(train_x.shape)
    return x,y

# 获取train,val,test路径下的图像名
print("正在加载数据集... ...")
train_img_list = getPathImg(os.path.join(root_dir,'train'),[])
val_img_list = getPathImg(os.path.join(root_dir,'val'),[])
test_img_list = getPathImg(os.path.join(root_dir,'test'),[])

# 获取图像矩阵，标签信息
print("正在加载图像... ...")
train_x,train_y = getImg(train_img_list)
val_x,val_y = getImg(val_img_list)
test_x,test_y = getImg(test_img_list)

# 转换为tensor
train_x,train_y = to_tensor(train_x,train_y)
val_x,val_y = to_tensor(val_x,val_y)
test_x,test_y = to_tensor(test_x,test_y)

# DataLoader加载数据集
print("正在读取数据集... ...")
data_train = DataLoader(Data.TensorDataset(train_x,train_y),batch_size=n_batch_size,shuffle=True)
data_val = DataLoader(Data.TensorDataset(val_x,val_y),batch_size=n_batch_size,shuffle=True)
data_test = DataLoader(Data.TensorDataset(test_x,test_y),batch_size=n_batch_size,shuffle=True)

# module_v1 = module_v1.Net()
# module_v1 = module_v2.LeNet()
# module_v1 = module_v3.Net2()
module_v1 = module_v4.AlexNet()

# 定义优化器
optimizer = Adam(module_v1.parameters(),lr = n_lr,betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
# optimizer = SGD(module_v1.parameters(),lr = 1e-5)
# optimizer = SGD(module_v1.parameters(), lr=n_lr)

# 定义loss函数
criterion = nn.CrossEntropyLoss()

# 检查gpu
# 检查gpu是否可用，否则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("use device:",device)

module_v1.to(device)
criterion.to(device)

# print("使用的模型如下：")
# print(module_v1)
train_loss = []
val_loss = []
for epoch in tqdm(range(1,n_epochs)):
    correct = 0
    total = 0
    loss_train = []
    loss_val = []
    module_v1.train()  # 训练开始
    for inputs, labels in data_train:
        # 把数据放在gpu上
        inputs = Variable(inputs,requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()  # 梯度置0
        output = module_v1(inputs)

        # 将损失函数softmax一下
        probs = nn.Softmax(dim=1)(output)
        # probs = torch.max(probs,1)[1]
        loss = criterion(probs, labels)  # 计算损失
        loss_train.append(loss.item())

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 输出正确率
        total += labels.size(0)
        _, preds_tensor = torch.max(output, 1)
        correct += np.squeeze((preds_tensor == labels).sum().cpu().numpy())
        # print(correct / total)

    module_v1.eval()  # 验证开始
    for inputs, labels in data_val:

        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        output = module_v1(inputs)
        loss = criterion(output, labels)
        loss_val.append(loss.item())

    train_loss.append(np.mean(loss_train))
    val_loss.append(np.mean(loss_val))
    print("Epoch:{}, Training Loss:{}, Valid Loss:{}".format(epoch, np.mean(train_loss), np.mean(val_loss)))
    print("Accuracy : {} %".format(correct / total))
print("======= Training Finished ! =========")

plt.plot(train_loss,label = 'Training loss')
plt.plot(val_loss,label = 'Validation loss')
plt.legend()
plt.show()

print("Testing Begining ... ")  # 模型测试
total = 0
correct = 0
for i, data_tuple in enumerate(data_test, 0):
    inputs, labels = data_tuple
    inputs = Variable(inputs, requires_grad=True).to(device)
    labels = Variable(labels).to(device)

    output = module_v1(inputs)
    _, preds_tensor = torch.max(output, 1)

    total += labels.size(0)
    correct += np.squeeze((preds_tensor == labels).sum().cpu().numpy())
print("Accuracy : {} %".format(correct / total))
