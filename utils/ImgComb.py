# combine two images from two folders

import os
import cv2
import numpy as np


def combine_images(folder_path1, folder_path2, save_path):
    if not os.path.exists(folder_path1) or not os.path.exists(folder_path2):
        print('folder not exists')
        return
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for image in os.listdir(folder_path1):
        if image.endswith('.jpg') or image.endswith('.png'):
            print(image)
            img1 = cv2.imread(os.path.join(folder_path1, image))
            img2 = cv2.imread(os.path.join(folder_path2, image))
            img = np.concatenate((img1, img2), axis=1)
            cv2.imwrite(os.path.join(save_path, image), img)

if __name__ == '__main__':
    folder_path1 = '../DehazeFormer/data/BeDDE/test/hazy'
    folder_path2 = '../DehazeFormer/data/BeDDE/test/dehazing'
    save_path = '../DehazeFormer/data/BeDDE/test/combined'
    combine_images(folder_path1, folder_path2, save_path)
    print('done')