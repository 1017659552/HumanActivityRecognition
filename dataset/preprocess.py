import os
import cv2
import numpy as np
import torch
import PIL.Image as Image
from torchvision import transforms as transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset
from myparam import Param

from sklearn.cluster import KMeans
import shutil

class MyDataset(Dataset):
    # preprocess_cut 是否执行视频分割帧操作
    # preprocess_rmbg 是否执行删除背景操作
    def __init__(self,preprocess_cut=False,preprocess_rmbg=False,get3d_data = False,split='train',clip_len=16):
        self.root_dir, self.output_dir = Param.dataset_dir(self)
        self.crop_size,self.frame_width,self.frame_height = Param.img_size(self)
        self.clip_len = clip_len
        self.get3d_data = get3d_data
        folder = os.path.join(self.output_dir, split)

        # 数据预处理—-视频分割帧
        if(preprocess_cut):
            print('正在执行视频分割帧操作...')
            self.preprocess_cut()

        # 数据预处理--清理图片
        if(preprocess_rmbg):
            print('正在执行删除背景图片操作...')
            self.preprocess_rmbg()

        # 随机翻转
        # for label in sorted(os.listdir(folder)):  # 每个类别
        #     for fname in os.listdir(os.path.join(folder, label)):  # 打开每个类别的文件夹
        #         for imgs in os.listdir(os.path.join(folder, label, fname)):
        #             im = Image.open(os.path.join(folder, label, fname,imgs))
        #             new_im = transforms.RandomHorizontalFlip(p=0.5)(im)  # p表示概率
        #             new_im.save(os.path.join(folder,label,fname,imgs+'a.jpg'))
        #             new_im = transforms.RandomVerticalFlip(p=0.5)(im)
        #             new_im.save(os.path.join(folder,label,fname,imgs+'b.jpg'))

        self.fnames,labels = [],[]
        if(get3d_data):
            for label in sorted(os.listdir(folder)): # 每个类别
                for fname in os.listdir(os.path.join(folder,label)): # 打开每个类别的文件夹
                    img_path = os.path.join(folder, label, fname)
                    num = os.listdir(img_path)
                    if (len(num) < self.clip_len):
                        shutil.rmtree(img_path)
                    else:
                        self.fnames.append(os.path.join(folder,label,fname))
                        labels.append(label)
        else:
            for label in sorted(os.listdir(folder)):  # 每个类别
                for fname in os.listdir(os.path.join(folder, label)):  # 打开每个类别的文件夹
                    for imgs in os.listdir(os.path.join(folder, label,fname)):
                        self.fnames.append(os.path.join(folder, label, fname,imgs))
                        labels.append(label)

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # 将类别转换为索引 [0 0 0 0 0 1 1 1 1 ......]
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)


    def preprocess_cut(self):
        # 自动创建文件夹
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir,'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))
            os.mkdir(os.path.join(self.output_dir, 'val'))

        # 将数据分割为train/val/test三部分
        for category in tqdm(os.listdir(self.root_dir)):
            # 获取每一类别下的所有视频
            category_path = os.path.join(self.root_dir,category)
            video_files = [name for name in os.listdir(category_path)]
            np.random.shuffle(video_files)
            # 按0.8-0.2分割数据集
            train_and_val,test = train_test_split(video_files,test_size=0.2,shuffle=True)
            train,val = train_test_split(train_and_val,test_size = 0.2,shuffle=True)
            #设置路径
            train_dir = os.path.join(self.output_dir,'train',category)
            val_dir = os.path.join(self.output_dir, 'val', category)
            test_dir = os.path.join(self.output_dir, 'test', category)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            # 按帧分割视频
            for video in train:
                self.process_video(video,category,train_dir)
            for video in val:
                self.process_video(video,category,val_dir)
            for video in test:
                self.process_video(video,category,test_dir)

    def process_video(self,video,category,save_dir):
        # 为每个视频创建文件夹，存放截取出来的帧
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir,video_filename)):
            os.mkdir(os.path.join(save_dir,video_filename))

        # 用openCV读取视频
        capture = cv2.VideoCapture(os.path.join(self.root_dir,category,video))

        # 获取视频帧数
        # 确保分割的图像至少有16帧
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        EXTRACT_FREQUENCY = 4 #每隔4帧取一帧
        # if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
        #     EXTRACT_FREQUENCY -= 1
        #     if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
        #         EXTRACT_FREQUENCY -= 1
        #         if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
        #             EXTRACT_FREQUENCY -= 1
        count = 0
        i=0
        retaining = True
        # 裁剪帧并写入文件夹里
        while (count < frame_count and retaining):
            retaining,frame = capture.read()
            if frame is None:
                continue
            if count % EXTRACT_FREQUENCY == 0:

                # 图片剪裁为crop_size大小的正方形
                y0 = int((self.frame_height-self.crop_size)/2)
                x0 = int((self.frame_width-self.crop_size)/2)
                frame = frame[y0:self.crop_size+y0,x0:self.crop_size+x0]

                #图片存入文件夹
                cv2.imwrite(filename=os.path.join(save_dir,video_filename,'000{}.jpg'.format(str(i))),img=frame)
                i = i + 1
            count = count + 1
        capture.release()

    def preprocess_rmbg(self):
        list = ['train','val','test']
        for dir in tqdm(list):
            train_dir = os.path.join(self.output_dir, dir)
            for category in tqdm(os.listdir(train_dir)):
                if category in ['Walking', 'Jogging', 'Running']:
                    train_category_dir = os.path.join(train_dir, category)
                    for unity in os.listdir(train_category_dir):
                        category_dir = os.path.join(train_category_dir, unity)
                        # 按目录顺序删除
                        self.delete_background(category_dir)

    def delete_background(self,category_dir):
        img_features = []
        for item in os.listdir(category_dir):
            img_path = os.path.join(category_dir,item)
            # 读取图片
            img = np.array(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
            # 直方图矩阵
            hist = np.array(cv2.calcHist([img], [0], None, [256], [0, 256]))
            # 找到最大值和最小值
            max_index = hist.argmax()
            hist_tmp = hist[max_index:]
            min_index = hist_tmp.argmin()
            zero_max = max_index + min_index
            zero_min = (hist != 0).argmax()
            diff_value = zero_max - zero_min
            # 创建特征矩阵
            feature_item = [diff_value,hist.var(),hist.max()]
            img_features.append(feature_item)
        img_features = np.array(img_features)

        #创建分类器
        estimator = KMeans(n_clusters=2)  # 构造聚类器
        estimator.fit(img_features)  # 聚类
        label_pred = np.array(estimator.labels_)  # 获取聚类标签
        # 通过最大值和最小值来判断分类标记表示有人还是没有人
        index_1 = label_pred.argmax()
        index_0 = label_pred.argmin()
        if (img_features[index_1][0] > img_features[index_0][0]):
            flag = 1  # 1表示有人
        else:
            flag = 0  # 0表示有人
        #根据预测结果删除
        img_list = os.listdir(category_dir)
        for i in range(len(img_list)):
            if flag == 1:
                if label_pred[i] == 0:
                    os.remove(os.path.join(category_dir,img_list[i]))
                    print('成功删除！ path = '+ os.path.join(category_dir,img_list[i]))
            else:
                if label_pred[i] == 1:
                    os.remove(os.path.join(category_dir,img_list[i]))
                    print('成功删除！ path = ' + os.path.join(category_dir, img_list[i]))


########################################33
    def __len__(self):
        return  len(self.fnames)

    # 按规定重写
    # 重新定义返回结构 buffer , [类别，视频名]
    def __getitem__(self, item):
        if(self.get3d_data):
            buffer = self.load_frames(self.fnames[item]) # 加载视频帧
            buffer = self.crop(buffer,self.clip_len,self.crop_size) # 选取帧片段并进行裁剪
            labels = np.array(self.label_array[item])   # 加载标签
            buffer = self.to_tensor(buffer)
            return torch.from_numpy(buffer), torch.from_numpy(labels)
        else:
            buffer = self.fnames[item]  # 加载视频帧
            labels = self.label_array[item]

            buffer_img = np.array(cv2.imread(buffer, cv2.IMREAD_GRAYSCALE))
            buffer_img = buffer_img /255.0
            buffer_img = buffer_img.astype(np.float64)

            buffer_img = buffer_img.reshape(1,self.crop_size, self.crop_size)
            inf = []
            inf.append(labels)
            inf.append(os.path.abspath(os.path.dirname(buffer)).split('\\')[-1])

            return torch.Tensor(buffer_img), inf


    def load_frames(self,frame_dir):
        frames = sorted([os.path.join(frame_dir,img) for img in os.listdir(frame_dir)])
        frame_count = len(frames)
        ######################
        # buffer = np.empty((frame_count,self.crop_size,self.crop_size),np.dtype('float32'))
        buffer = np.empty((frame_count, self.crop_size,self.crop_size,3), np.dtype('float32'))
        # print(buffer.shape)
        for i,frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name,cv2.IMREAD_UNCHANGED)) # 读取灰色图像 cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
            frame = frame / 255.0
            frame = frame.astype(np.float64)
            # print(frame.shape)
            buffer[i] = frame
        # print(buffer.shape)
        return buffer

    def crop(self,buffer,clip_len,crop_size):
        # time_index = np.random.randint(buffer.shape[0]-clip_len) # 保证能取到clip_len帧
        # print(len(buffer))
        buffer = buffer[0:clip_len]

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3,0, 1, 2))

    def horizontal_flip(image):
        HF = transforms.RandomHorizontalFlip()
        hf_image = HF(image)
        return hf_image