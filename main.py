from tqdm import tqdm
# import pandas as pd
import  numpy as np
import os
import cv2
import matplotlib.pyplot as plt

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
root_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess'
labels = ['boxing','handclapping','handwaving','jogging','running','walking']
img_width = 160
img_height = 120

train_img_name = getPathImg(os.path.join(root_dir,'val'),[])

print("(1)正在读取train图像... ...")
train_imgs = []
train_labels = []
for img_name in tqdm(train_img_name):
    # 加载图片
    img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE) #读取灰度图像
    img = img/255.0 #归一化像素值
    img = img.astype(np.float32)
    train_imgs.append(img)
    # 加载标签
    label = img_name.split('\\')[-2].split('_')[1]
    train_labels.append(labels.index(label))

assert len(train_labels)==len(train_imgs),'图像和标签数量不一致，退出程序！'

# 将读取的图像挑4张可视化
train_x = np.array(train_imgs)
train_y = np.array(train_labels)
i = 0
plt.figure(figsize=(30,30))
plt.subplot(221),plt.imshow(train_x[i],cmap = 'gray')
plt.subplot(222),plt.imshow(train_x[i+25],cmap = 'gray')
plt.subplot(223),plt.imshow(train_x[i+50],cmap = 'gray')
plt.subplot(224),plt.imshow(train_x[i+75],cmap = 'gray')
plt.show()

