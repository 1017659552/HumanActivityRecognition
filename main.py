from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
import cv2
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam,SGD
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset.preprocess import MyDataset

import network.module_v1 as module_v1
import network.module_v2 as module_v2
import network.module_v3 as module_v3
import network.module_v4 as module_v4
import network.module_v5 as module_v5

########### 参数设置 #################
root_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess_v3'
# root_dir = '/home/mist/old/KTH_preprocess_v3'
labels = ['boxing','handclapping','handwaving','jogging','running','walking']
n_epochs = 6
n_batch_size = 128
n_lr = 1e-3

img_width = 120
img_height = 120
crop_size = 120
####################################
print(root_dir)

# preprocess.MyDataset()

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
# print("正在加载数据集... ...")
# train_img_list = getPathImg(os.path.join(root_dir,'train'),[])
# val_img_list = getPathImg(os.path.join(root_dir,'val'),[])
# test_img_list = getPathImg(os.path.join(root_dir,'test'),[])
#
# # 获取图像矩阵，标签信息
# print("正在加载图像... ...")
# train_x,train_y = getImg(train_img_list)
# val_x,val_y = getImg(val_img_list)
# test_x,test_y = getImg(test_img_list)
#
# # 转换为tensor
# train_x,train_y = to_tensor(train_x,train_y)
# val_x,val_y = to_tensor(val_x,val_y)
# test_x,test_y = to_tensor(test_x,test_y)

# DataLoader加载数据集
print("正在读取数据集... ...")
# data_train = DataLoader(Data.TensorDataset(train_x,train_y),batch_size=n_batch_size,shuffle=True)
# data_val = DataLoader(Data.TensorDataset(val_x,val_y),batch_size=n_batch_size,shuffle=True)
# data_test = DataLoader(Data.TensorDataset(test_x,test_y),batch_size=n_batch_size,shuffle=True)
# 读取数据集
data_train = DataLoader(MyDataset(split='train', clip_len=16),batch_size=n_batch_size, shuffle=True)
data_test = DataLoader(MyDataset(split='test', clip_len=16), batch_size=n_batch_size, shuffle=True)
data_val = DataLoader(MyDataset(split='val', clip_len=16), batch_size=n_batch_size, shuffle=True)
# 读取标签目录
catalog_train = pd.read_table('./dataset/catalog_train.txt',sep=',')
catalog_test = pd.read_table('./dataset/catalog_test.txt',sep=',')
catalog_val = pd.read_table('./dataset/catalog_val.txt',sep=',')

# module_v1 = module_v1.Net()
# module_v1 = module_v2.LeNet()
# module_v1 = module_v3.Net2()
module_v1 = module_v4.AlexNet()
# module_v1 = module_v5.C3D()

# 定义优化器
# optimizer = Adam(module_v1.parameters(),lr = n_lr,betas=(0.9, 0.99), eps=1e-06, weight_decay=1e-3) # 加了这个，全部预测为同一个类
optimizer = SGD(module_v1.parameters(),lr = n_lr)
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
# val_loss = []
train_acc = []
val_acc = []
# train_loss_show = []
# val_loss_show = []
for epoch in tqdm(range(1,n_epochs)):
    correct_train = 0
    correct_val = 0
    total_train = 0
    total_val = 0
    loss_train = []
    # loss_val = []

    module_v1.train()  # 训练开始
    for inputs, inf in data_train:
        # 创建投票空表格
        vote_table = pd.DataFrame(columns=('video_name', '0', '1', '2', '3', '4', '5'))
        vote_table = vote_table.iloc[1:7].astype(int)

        # 把数据放在gpu上
        inputs = Variable(inputs,requires_grad=True).to(device)
        # labels_tensor = torch.Tensor(labels[0])
        label = Variable(inf[0]).to(device,dtype=torch.int64)

        optimizer.zero_grad()  # 梯度置0
        output = module_v1(inputs)

        # 将结果softmax一下，转换为概率
        probs = nn.Softmax(dim=1)(output)
        probs_cpu = probs.cpu().detach().numpy()
        probs_label = torch.max(probs, 1)[1].cpu().numpy()

        # 按照概率进行投票
        for i in range(0,len(probs_label)):
            video_name = inf[1][i]
            index = vote_table[vote_table['video_name']==video_name].index.tolist()
            if len(index)==0:
                # print('b')
                num_row = vote_table.shape[0]
                vote_table.loc[num_row] = [video_name,probs_cpu[i][0],probs_cpu[i][1],probs_cpu[i][2],probs_cpu[i][3],probs_cpu[i][4],probs_cpu[i][5]]
                # vote_table.iloc[num_row,probs_label[i]+1] = 1
            else:
                tmp = index[0]
                vote_table.iloc[tmp,1] = (vote_table.iloc[tmp,1]+probs_cpu[tmp][0])/2
                vote_table.iloc[tmp, 2] = (vote_table.iloc[tmp, 2] + probs_cpu[tmp][1]) / 2
                vote_table.iloc[tmp, 3] = (vote_table.iloc[tmp, 3] + probs_cpu[tmp][2]) / 2
                vote_table.iloc[tmp, 4] = (vote_table.iloc[tmp, 4] + probs_cpu[tmp][3]) / 2
                vote_table.iloc[tmp, 5] = (vote_table.iloc[tmp, 5] + probs_cpu[tmp][4]) / 2
                vote_table.iloc[tmp, 6] = (vote_table.iloc[tmp, 6] + probs_cpu[tmp][5]) / 2

                # vote_table.iloc[tmp, 2] += probs_cpu[tmp][1]
                # # vote_table.iloc[tmp, 3] += probs_cpu[tmp][2]
                # # vote_table.iloc[tmp, 4] += probs_cpu[tmp][3]
                # # vote_table.iloc[tmp, 5] += probs_cpu[tmp][4]
                # # vote_table.iloc[tmp, 6] += probs_cpu[tmp][5]

                # tmp = vote_table.iloc[index[0],probs_label[i]+1]
                # vote_table.iloc[num_row,probs_label[i]+1] = tmp+1

        vote_table['pred_class'] = vote_table.iloc[:,1:7].idxmax(axis=1)
        # pred_label = vote_table['class'].values
        table_merge = pd.merge(vote_table,catalog_train,on='video_name')
        pred_label = table_merge['pred_class'].astype(int).values
        true_label = table_merge['class'].values

        acc = accuracy_score(true_label,pred_label)
        train_acc.append(acc)
        print('Training acc : '+ str(acc))

        probs_vote = table_merge.iloc[:,1:7].values
        probs_vote_tensor = torch.tensor(probs_vote).requires_grad_()
        true_vote_tensor = torch.tensor(true_label,dtype=torch.long)

        loss = criterion(probs_vote_tensor, true_vote_tensor)  # 计算损失
        loss_train.append(loss.item())

        # loss = criterion(probs, label)  # 计算损失
        # loss_train.append(loss.item())
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 输出正确率
        # total_train += label.size(0)
        # _, preds_tensor = torch.max(output, 1)
        # correct_train += np.squeeze((preds_tensor == label).sum().cpu().numpy())
        # train_acc.append(np.mean(correct_train / total_train))
        # print(np.mean(correct_train / total_train))

    # module_v1.eval()  # 验证开始
    # for inputs, labels in data_val:
    #
    #     inputs = Variable(inputs, requires_grad=True).to(device)
    #     labels = Variable(labels).to(device,dtype=torch.int64)
    #     optimizer.zero_grad()  # 梯度置0
    #     output = module_v1(inputs)
    #
    #     probs = nn.Softmax(dim=1)(output)
    #     loss = criterion(probs, labels)
    #     loss_val.append(loss.item())
    #
    #     total_val += labels.size(0)
    #     _, preds_tensor = torch.max(output, 1)
    #     correct_val += np.squeeze((preds_tensor == labels).sum().cpu().numpy())
    #     val_acc.append(np.mean(correct_val / total_val))

    # train_loss.append(np.mean(loss_train))
    # val_loss.append(np.mean(loss_val))
    # train_loss_show.append(np.mean(train_loss))
    # val_loss_show.append(np.mean(val_loss))
    # print("Epoch:{}, Training Loss:{}, Valid Loss:{}".format(epoch, np.mean(loss_train), np.mean(loss_val)))
    # print("Accuracy : {} %".format(correct / total))
print("======= Training Finished ! =========")

plt.plot(train_loss,label = 'Training loss')
# plt.plot(val_loss,label = 'Validation loss')
plt.legend()
plt.show()

plt.plot(train_acc,label = 'Training acc')
# plt.plot(val_acc,label = 'Validation acc')
plt.legend()
plt.show()

# # plt.plot(train_acc,label = 'Training acc')
# plt.plot(val_acc,label = 'Validation acc')
# plt.legend()
# plt.show()

print("Testing Begining ... ")  # 模型测试
test_acc = []
for inputs, inf in data_test:
    # 创建投票空表格
    vote_table = pd.DataFrame(columns=('video_name', '0', '1', '2', '3', '4', '5'))
    vote_table = vote_table.iloc[1:7].astype(int)

    # 把数据放在gpu上
    inputs = Variable(inputs, requires_grad=True).to(device)
    # labels_tensor = torch.Tensor(labels[0])
    label = Variable(inf[0]).to(device, dtype=torch.int64)

    optimizer.zero_grad()  # 梯度置0
    output = module_v1(inputs)

    # 将结果softmax一下，转换为概率
    probs = nn.Softmax(dim=1)(output)
    probs_cpu = probs.cpu().detach().numpy()
    probs_label = torch.max(probs, 1)[1].cpu().numpy()

    # 按照概率进行投票
    for i in range(0, len(probs_label)):
        video_name = inf[1][i]
        index = vote_table[vote_table['video_name'] == video_name].index.tolist()
        if len(index) == 0:
            # print('b')
            num_row = vote_table.shape[0]
            vote_table.loc[num_row] = [video_name, probs_cpu[i][0], probs_cpu[i][1], probs_cpu[i][2], probs_cpu[i][3],
                                       probs_cpu[i][4], probs_cpu[i][5]]
            # vote_table.iloc[num_row,probs_label[i]+1] = 1
        else:
            tmp = index[0]
            vote_table.iloc[tmp, 1] = (vote_table.iloc[tmp, 1] + probs_cpu[tmp][0]) / 2
            vote_table.iloc[tmp, 2] = (vote_table.iloc[tmp, 2] + probs_cpu[tmp][1]) / 2
            vote_table.iloc[tmp, 3] = (vote_table.iloc[tmp, 3] + probs_cpu[tmp][2]) / 2
            vote_table.iloc[tmp, 4] = (vote_table.iloc[tmp, 4] + probs_cpu[tmp][3]) / 2
            vote_table.iloc[tmp, 5] = (vote_table.iloc[tmp, 5] + probs_cpu[tmp][4]) / 2
            vote_table.iloc[tmp, 6] = (vote_table.iloc[tmp, 6] + probs_cpu[tmp][5]) / 2

    vote_table['pred_class'] = vote_table.iloc[:, 1:7].idxmax(axis=1)
    # pred_label = vote_table['class'].values
    table_merge = pd.merge(vote_table, catalog_test, on='video_name')
    pred_label = table_merge['pred_class'].astype(int).values
    true_label = table_merge['class'].values

    acc = accuracy_score(true_label, pred_label)
    train_acc.append(acc)
    print('Training acc : ' + str(acc))

    probs_vote = table_merge.iloc[:, 1:7].values
    probs_vote_tensor = torch.tensor(probs_vote).requires_grad_()
    true_vote_tensor = torch.tensor(true_label,dtype=torch.long)

    loss = criterion(probs_vote_tensor, true_vote_tensor)  # 计算损失

    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
print("Accuracy : {} %".format(np.mean(test_acc)))

plt.plot(test_acc,label = 'Testing acc')
# plt.plot(val_acc,label = 'Validation acc')
plt.legend()
plt.show()
