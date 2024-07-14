# read images in folder and resize them to 256x256

import os
import cv2
import numpy as np


def resize_images(folder_path, save_path, size=(256, 256)):
    if not os.path.exists(folder_path):
        print('folder not exists')
        return
    
    for image in os.listdir(folder_path):
        if image.endswith('.jpg') or image.endswith('.png'):
            print(image)
            img = cv2.imread(os.path.join(folder_path, image))
            img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(save_path, image), img)

if __name__ == '__main__':
    folder_path = '../DehazeFormer/data/BeDDE/test/hazy'
    save_path = '../DehazeFormer/data/BeDDE/test/hazy'
    resize_images(folder_path, save_path)
    folder_path = '../DehazeFormer/data/BeDDE/test/GT'
    save_path = '../DehazeFormer/data/BeDDE/test/GT'
    resize_images(folder_path, save_path)
    print('done')