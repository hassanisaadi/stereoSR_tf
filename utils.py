import gc
import os
import sys

import numpy as np
import tensorflow as tf
#from PIL import Image
import cv2
import h5py


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

class train_data():
    def __init__(self, filepath='./data/data_da1_p33_s24_b128_tr60.hdf5'):
        self.filepath = filepath
        assert '.hdf5' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = h5py.File(self.filepath, "r")
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")

def load_data(filepath='./data/data_da1_p33_s24_b128_tr60.hdf5'):
    return train_data(filepath=filepath)

def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = cv2.imread(filelist, flags=cv2.IMREAD_COLOR)
        im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)    # Y Cr Cb = [0,1,2]
        return im_ycrcb
    data = []
    for file in filelist:
        im = cv2.imread(file, flags=cv2.IMREAD_COLOR)
        im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
        data.append(im_ycrcb)
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None, iter_num=0, idx=0):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    #cat_image = cv2.cvtColor(cat_image, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite(filepath, cat_image.astype('uint8'))

    #fpath_gt  = './eval_results_cpu/test%d_%d_gt.png' % (idx, iter_num)
    #fpath_in  = './eval_results_cpu/test%d_%d_in.png' % (idx, iter_num)
    #fpath_out = './eval_results_cpu/test%d_%d_out.png' % (idx, iter_num)
    #cv2.imwrite(fpath_gt, ground_truth.astype('uint8'))
    #cv2.imwrite(fpath_in, noisy_image.astype('uint8'))
    #cv2.imwrite(fpath_out, clean_image.astype('uint8'))

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


