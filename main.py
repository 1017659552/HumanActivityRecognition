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


def getPathImg(path,files):
    '''
    递归获取目录下所有文件
    :param path: 根目录
    :param files: 存放文件名的数组
    :return: files
    '''
    file_list = os.listdir(path)
    for file in file_list:
        path_cur = os.path.join(path,file)
        if os.path.isdir(path_cur):
            getPathImg(path_cur,files)
        else:
            files.append(os.path.join(path_cur))


    return files

'''
参数设置
'''
# root_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess'
root_dir = '/home/mist/KTH_preprocess'
labels = ['boxing','handclapping','handwaving','jogging','running','walking']
n_epochs = 30
img_width = 120
img_height = 120
crop_size = 120

train_img_name = getPathImg(os.path.join(root_dir,'val'),[])
random.shuffle(train_img_name)  # 乱序

print("(1)正在读取train图像... ...")
train_imgs = []
train_labels = []
for img_name in tqdm(train_img_name):
    # 加载图片
    img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)) #读取灰度图像
    # 查看灰度直方图
    # print(img_name.split('\\')[5])
    # if(img_name.split('\\')[5]=='Jogging'):
    #     hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #     print(hist)
    #     plt.hist(img.ravel(), 256, [0,256])
    #     plt.show()
    #     print(hist)
    # 裁剪图片
    y0 = int((img_height-crop_size)/2)
    x0 = int((img_width-crop_size)/2)
    img = img[y0:crop_size+y0,x0:crop_size+x0]

    img = img/255.0 #归一化像素值
    img = img.astype(np.float32)
    train_imgs.append(img)
    # 加载标签
    # label = img_name.split('\\')[-2].split('_')[1] ######################本地
    label = img_name.split('/')[-2].split('_')[1] #######################远程
    train_labels.append(labels.index(label))

assert len(train_labels)==len(train_imgs),'图像和标签数量不一致，退出程序！'
train_imgs = np.array(train_imgs)
train_labels = np.array(train_labels)

# 将读取的图像挑4张可视化
# train_x = np.array(train_imgs)
# train_y = np.array(train_labels)
# i = 0
# plt.figure(figsize=(30,30))
# plt.subplot(221),plt.imshow(train_x[i],cmap = 'gray')
# plt.subplot(222),plt.imshow(train_x[i+25],cmap = 'gray')
# plt.subplot(223),plt.imshow(train_x[i+50],cmap = 'gray')
# plt.subplot(224),plt.imshow(train_x[i+75],cmap = 'gray')
# plt.show()

# 创建验证集对图像进行预处理
train_x,val_x,train_y,val_y = train_test_split(train_imgs,train_labels,test_size=0.2,shuffle=True)

# 将训练集转换成torch张量
train_x_len = len(train_x)
train_x = train_x.reshape(train_x_len,1,crop_size,crop_size)
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
# print(train_x.shape)
# 将验证集转换成torch张量
val_x_len = len(val_x)
val_x = val_x.reshape(val_x_len,1,crop_size,crop_size)
val_x = torch.from_numpy(val_x)
val_y = torch.from_numpy(val_y)
# print(val_x.shape)

# DataLoader加载数据集
data_train = DataLoader(Data.TensorDataset(train_x,train_y),batch_size=128,shuffle=True)
data_test = DataLoader(Data.TensorDataset(val_x,val_y),batch_size=128,shuffle=True)

module_v1 = module_v1.Net()
# module_v1 = module_v2.LeNet()
# 定义优化器
# optimizer = Adam(module_v1.parameters(),lr = 0.001)
# optimizer = SGD(module_v1.parameters(),lr = 1e-5)
optimizer = SGD(module_v1.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

# 定义loss函数
criterion = nn.CrossEntropyLoss()
# 检查gpu
# 检查gpu是否可用，否则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("use device:",device)

# if torch.cuda.is_available():
#     module_v1.to(device)
#     criterion.to(device)

print("使用的模型如下：")
# print(module_v1)

for epoch in range(1,n_epochs):
    train_loss = []
    correct = 0
    total = 0
    module_v1.train()  # 训练开始
    for data, target in data_train:
        optimizer.zero_grad()  # 梯度置0
        output = module_v1(data)
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_loss.append(loss.item())
        # 输出正确率
        total += target.size(0)
        _, preds_tensor = torch.max(output, 1)
        correct += np.squeeze((preds_tensor == target).sum().numpy())
        print(correct / total)

    # net.eval()  # 验证开始
    # for data, target in validloader:
    #     output = net(data)
    #     loss = loss_function(output, target)
    #     valid_loss.append(loss.item())

    # print("Epoch:{}, Training Loss:{}, Valid Loss:{}".format(epoch, np.mean(train_loss), np.mean(valid_loss)))
    print("Epoch:{}, Training Loss:{}".format(epoch, np.mean(train_loss)))
print("======= Training Finished ! =========")

print("Testing Begining ... ")  # 模型测试
total = 0
correct = 0
for i, data_tuple in enumerate(data_test, 0):
    data, labels = data_tuple
    output = module_v1(data)
    _, preds_tensor = torch.max(output, 1)

    total += labels.size(0)
    correct += np.squeeze((preds_tensor == labels).sum().numpy())
print("Accuracy : {} %".format(correct / total))






#
# def cal_accuracy(model, x_test, y_test, samples=10000):
#
#     y_pred = model(x_test[:samples])
#     y_pred_ = list(map(lambda x: np.argmax(x), y_pred.data.numpy()))
#     acc = sum(y_pred_ == y_test.numpy()[:samples]) / samples
#     return acc
#
# def train(epoch):
#     # module_v1.train()
#     # tr_loss = 0
#     # #获取训练集和验证集
#     # x_train,y_train = Variable(train_x),Variable(train_y)
#     # x_val,y_val = Variable(val_x),Variable(val_y)
#     # #这里转换标签的格式，否则会报错 expected scalar type Long but found Int
#     # y_train = y_train.type(torch.LongTensor)
#     # y_val = y_val.type(torch.LongTensor)
#     #
#     # # 转换为gpu格式
#     # # if torch.cuda.is_available():
#     # #     x_train = x_train.to(device)
#     # #     y_train = y_train.to(device)
#     # #     x_val = x_val.to(device)
#     # #     y_val = y_val.to(device)
#     # # 清除梯度
#     # optimizer.zero_grad()
#     # # 预测训练与验证集
#     # output_train = module_v1(x_train)
#     # output_val = module_v1(x_val)
#     # # 计算验证集和训练集损失
#     # loss_train = criterion(output_train,y_train)
#     # loss_val = criterion(output_val,y_val)
#     # train_losses.append(loss_train)
#     # val_losses.append(loss_val)
#     # # 更新权重
#     # loss_train.backward()
#     # optimizer.step()
#     # tr_loss = loss_train.item()
#     # if(epoch%2==0):
#     #     #输出验证集loss
#     #     print('Epoch:',epoch+1,'  loss:',loss_val)
#     ##################################################
#     # 另一种写法
#     acc_data = []
#     loss_data = []
#     for index,(x_data,y_data) in enumerate(data_train):
#         prediction = module_v1(torch.unsqueeze(x_data,dim = 1))
#         loss = criterion(prediction,y_data)
#         print('No.%s,loss=%.3f' % (index + 1, loss.data.numpy()))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_val = loss.data.numpy()
#         if epoch==0:
#             softmax = torch.exp(prediction)
#             prob = list(softmax.numpy())
#             predictions = np.argmax(prob,axis = 1)
#             acc = accuracy_score(y_data,predictions)
#             acc_data.append(acc)
#             loss_data.append(loss_val)
#         print('No.%s,loss=%.3f' % (epoch + 1, loss_val))
#         print('acc = ',acc)
#
#
#
# # train_losses = []
# # val_losses = []
# for epoch in tqdm(range(n_epochs)):
#     train(epoch)
#
# acc=cal_accuracy(module_v1,torch.unsqueeze(val_x,dim=1),val_y,samples=10000)
#
# # plt.plot(train_losses,label = 'Training loss')
# # plt.plot(val_losses,label='Validation loss')
# # plt.legend()
# # plt.show()
# #
# # with torch.no_grad():
# #     output = module_v1(train_x)
# # softmax = torch.exp(output)
# # prob = list(softmax.numpy())
# # predictions = np.argmax(prob,axis = 1)
# #
# print(accuracy_score(train_y,predictions))