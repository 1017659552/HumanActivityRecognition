import os
import cv2
import numpy as np

crop_size = 120

def image2array(image_path):
    filenames = os.listdir(image_path)
    image_num = 10
    cube_num = int(image_num / 10)
    image_name = image_path + '\\' + filenames[0]
    Img = cv2.imread(image_name)
    Img = Img.reshape(1, crop_size, crop_size, 1)
    Img10 = Img
    for k in range(1, 10):
        image_name = image_path + '\\' + filenames[k]
        Img = cv2.imread(image_name)
        Img = Img.reshape(1, crop_size, crop_size, 1)
        Img10 = np.concatenate((Img10, Img), axis=0)
    cube = Img10.reshape(1, crop_size, crop_size, 1)
    for i in range(1, cube_num):
        index = 10 * i
        image_name = image_path + '\\' + filenames[index]
        Img = cv2.imread(image_name)
        Img = Img.reshape(1, crop_size, crop_size, 1)
        Img10 = Img
        for k in range(1, 10):
            image_name = image_path + '\\' + filenames[index + k]
            Img = cv2.imread(image_name)
            Img = Img.reshape(1, crop_size, crop_size, 1)
            Img10 = np.concatenate((Img10, Img), axis=0)
            Img10 = Img10.reshape(1, crop_size, crop_size, 1)
            cube = np.concatenate((cube, Img10), axis=0)
    return cube
