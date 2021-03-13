import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
print('\n')
# a = np.array([0,0,5,5,5,5,0,0,0])
# max_index = a.argmax()
# a_tmp = a[max_index:]
# min_index = a_tmp.argmin()
# print(min_index)
# aaa = max_index+min_index
# bbb = (a!=0).argmax()

root_dir = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\\Jogging\\person04_jogging_d4_uncomp'
file_list = os.listdir(root_dir)

img_num = []
for img_name in file_list:
    img_path = os.path.join(root_dir,img_name)
    img = np.array(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
    hist1 = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
    max_index = hist1.argmax()
    hist1_tmp = hist1[max_index:]
    min_index = hist1_tmp.argmin()
    zero_max = max_index+min_index
    zero_min = (hist1!=0).argmax()

    # plt.hist(img.ravel(), 256, [0,256])
    # plt.show()

    diff_value = zero_max-zero_min
    # print(diff_value)

    tmp = [diff_value,hist1.var(),hist1.max()]
    img_num.append(tmp)
    # print(img_name)
    # print(img_name+'------------->'+str(hist1.var(axis=0)))
img_num = np.array(img_num)

estimator = KMeans(n_clusters=2)#构造聚类器
estimator.fit(img_num)#聚类
label_pred = np.array(estimator.labels_) #获取聚类标签

index_1 = label_pred.argmax()
index_0 = label_pred.argmin()
if(img_num[index_1][0]>img_num[index_0][0]):
    flag = 1 #1表示有人
else:flag = 0
# centroids = estimator.cluster_centers_ #获取聚类中心
# inertia = estimator.inertia_ # 获取聚类准则的总和
# print(label_pred)
# print(label_pred)
for i in range(len(file_list)):
    if flag==1:
        if label_pred[i]==1:
            flag = '有人'
        else:flag = '无人'
    else:
        if label_pred[i]==1:
            flag = '无人'
        else:flag = '有人'
    print(file_list[i]+'------>'+flag)

#

# y = []
# for i in range(img_num.size):
#     y.append(0)
# plt.scatter(img_num,y)
# plt.show()


# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Jogging\\person08_jogging_d2_uncomp\\0000.jpg'#无人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist1 = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist1.var(axis=0))
# # print(hist1)
#
#
# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Jogging\\person08_jogging_d2_uncomp\\0003.jpg'#有人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist2 = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist2.var(axis=0))


#
# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Jogging\\person01_jogging_d4_uncomp\\0005.jpg'#有人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist2 = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist2.var(axis=0))
#
# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Jogging\\person01_jogging_d4_uncomp\\0008.jpg'#有人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist2 = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist2.var(axis=0))
# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Jogging\\person01_jogging_d4_uncomp\\00014.jpg'#无人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist2 = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist2.var(axis=0))

###########################
#
# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Walking\\person14_walking_d3_uncomp\\000128.jpg'# 无人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist.var(axis=0))
#
# img_name = 'D:\\SWUFEthesis\\data\\KTH_preprocess\\val\\Walking\\person14_walking_d3_uncomp\\000117.jpg' #有人
# img = np.array(cv2.imread(img_name,cv2.IMREAD_GRAYSCALE))
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()
# print(hist.var(axis=0))

