#! /usr/bin/env python2

import glob
import numpy as np
import h5py
import sys
import subprocess
import random
import os
from utils import *
import cv2

def generate_hdf5():
    SCALE = 2
    PARALLAX = 64
    PATCH_SIZE = 33
    STEP = PARALLAX
    STRIDE = 24
    BATCH_SIZE = 128
    DATA_AUG_TIMES = 1
    SRCDIR = '../Data/stereoSR_mb/'
    INTERPOLATION = cv2.INTER_CUBIC

    fpdata_tr  = sorted(glob.glob(SRCDIR + 'train/*.png'))
    numPics_tr = len(fpdata_tr) / 2

    DSTDIR = './data/'
    FDATA = DSTDIR + ('data_da%d_p%d_s%d_b%d_par%d_tr%d.hdf5' 
                    % (DATA_AUG_TIMES, PATCH_SIZE, STRIDE, BATCH_SIZE, PARALLAX, numPics_tr))
    SAVEPROB = 1
    CHKDIR = './data/chk/'

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    if not os.path.exists(CHKDIR):
        os.makedirs(CHKDIR)
    subprocess.check_call('rm -f {}/*'.format(CHKDIR), shell=True)
    
    count = 0
    for i in xrange(numPics_tr):
        fleft  = SRCDIR + ('train/im%d_0.png') % i
        img    = cv2.imread(fleft , flags=cv2.IMREAD_COLOR)  # BGR = [0,1,2]
        img_rs = cv2.resize(img   , (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=INTERPOLATION)
        img_s  = cv2.resize(img_rs, (int(img_rs.shape[1]/SCALE), int(img_rs.shape[0]/SCALE)), interpolation=INTERPOLATION)
        img_ss = cv2.resize(img_s , (int(img_s.shape[1]*SCALE), int(img_s.shape[0]*SCALE)), interpolation=INTERPOLATION)
        im_h, im_w, _ = img_ss.shape
        for x in range(0+STEP, (im_h-PATCH_SIZE), STRIDE):
            for y in range(0+STEP, (im_w-PATCH_SIZE), STRIDE):
                count += 1
    numPatches = count * DATA_AUG_TIMES
    #origin_patch_num = count * DATA_AUG_TIMES
    #if origin_patch_num % BATCH_SIZE != 0:
    #    numPatches = (origin_patch_num/BATCH_SIZE+1) * BATCH_SIZE
    #else:
    #    numPatches = origin_patch_num

    print("[*] Info ..")
    print("\t Number of train images = %d" % numPics_tr)
    print("\t Number of patches = %d" % numPatches)
    print("\t Patch size = %d" % PATCH_SIZE)
    print("\t Batch size = %d" % BATCH_SIZE)
    print("\t Number of batches = %d" % (numPatches/BATCH_SIZE))
    print("\t DATA_AUG_TIMES = %d" % DATA_AUG_TIMES)
    print("\t Source dir = %s" % SRCDIR)
    print("\t Dest dir = %s" % DSTDIR)
    print("\t Dest file = %s" % FDATA)
    sys.stdout.flush()

    shape_tr_in_lum  = (numPatches, PATCH_SIZE, PATCH_SIZE, PARALLAX+1)
    shape_tr_out_lum = (numPatches, PATCH_SIZE, PATCH_SIZE, 1)
    shape_tr_in_chr  = (numPatches, PATCH_SIZE, PATCH_SIZE, 2)
    shape_tr_out_chr = (numPatches, PATCH_SIZE, PATCH_SIZE, 3)

    hdfile = h5py.File(FDATA, mode = 'w')
    hdfile.create_dataset("tr_in_lum" , shape_tr_in_lum , np.uint8)
    hdfile.create_dataset("tr_out_lum", shape_tr_out_lum, np.uint8)
    hdfile.create_dataset("tr_in_chr" , shape_tr_in_chr , np.uint8)
    hdfile.create_dataset("tr_out_chr", shape_tr_out_chr, np.uint8)

    print("[*] Processing Train Images")
    
    c = 0
    for i in xrange(numPics_tr):
        print("\t Tr image [%2d/%2d]" % (i+1, numPics_tr))
        sys.stdout.flush()

        fleft  = SRCDIR + 'train/im%d_0.png' % i
        fright = SRCDIR + 'train/im%d_1.png' % i
        imgL  = cv2.imread(fleft , flags=cv2.IMREAD_COLOR) # BGR
        imgR  = cv2.imread(fright, flags=cv2.IMREAD_COLOR)

        imgL_rs = cv2.resize(imgL, (int(imgL.shape[1]/2), int(imgL.shape[0]/2)), interpolation=INTERPOLATION)
        imgR_rs = cv2.resize(imgR, (int(imgR.shape[1]/2), int(imgR.shape[0]/2)), interpolation=INTERPOLATION)

        imgL_s  = cv2.resize(imgL_rs, (int(imgL_rs.shape[1]/SCALE), int(imgL_rs.shape[0]/SCALE)), interpolation=INTERPOLATION)
        imgR_s  = cv2.resize(imgR_rs, (int(imgR_rs.shape[1]/SCALE), int(imgR_rs.shape[0]/SCALE)), interpolation=INTERPOLATION)

        imgL_ss = cv2.resize(imgL_s , (int(imgL_s.shape[1]*SCALE), int(imgL_s.shape[0]*SCALE)), interpolation=INTERPOLATION)
        imgR_ss = cv2.resize(imgR_s , (int(imgR_s.shape[1]*SCALE), int(imgR_s.shape[0]*SCALE)), interpolation=INTERPOLATION)

        imgL_rs_ycrcb = cv2.cvtColor(imgL_rs, cv2.COLOR_BGR2YCR_CB)    # Y Cr Cb = [0,1,2]
        
        imgL_ss_ycrcb = cv2.cvtColor(imgL_ss, cv2.COLOR_BGR2YCR_CB)
        imgR_ss_ycrcb = cv2.cvtColor(imgR_ss, cv2.COLOR_BGR2YCR_CB)

        im_h, im_w, _ = imgL_ss_ycrcb.shape
        for j in xrange(DATA_AUG_TIMES):
            for x in range(0+STEP, im_h-PATCH_SIZE, STRIDE):
                for y in range(0+STEP, im_w-PATCH_SIZE, STRIDE):
                    #mode = random.randint(0, 7)
                    mode = 0 ###!!! 
                    xx = np.zeros((1,PATCH_SIZE, PATCH_SIZE, PARALLAX+1))
                    xx[0,:,:,0] = data_augmentation(imgL_ss_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,0], mode)
                    pp = 0
                    for p in range(1,PARALLAX+1, 1):
                        xx[0,:,:,p] = data_augmentation(imgR_ss_ycrcb[x:x+PATCH_SIZE,y-pp:y+PATCH_SIZE-pp,0], mode)
                        pp += 1
                    yy = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                    yy[0,:,:,0] = data_augmentation(imgL_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,0], mode)
                    hdfile["tr_in_lum"][c, ...]  = xx
                    hdfile["tr_out_lum"][c, ...] = yy

                    x_chr = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 2))
                    y_chr = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
                    x_chr[0,:,:,:] = data_augmentation(imgL_ss_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,1:], mode)
                    y_chr[0,:,:,:] = data_augmentation(imgL_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,:] , mode)
                    hdfile["tr_in_chr"][c, ...] = x_chr
                    hdfile["tr_out_chr"][c, ...] = y_chr
                    if random.random() > SAVEPROB:
                        for p in range(0,PARALLAX+1,1):
                            cv2.imwrite(CHKDIR + ('%d_lum_in_%d.png' % (c, p)),xx[0,:,:,p])
                        cv2.imwrite(CHKDIR + ('%d_lum_out.png' % c), yy[0,:,:,:])
                        cv2.imwrite(CHKDIR + ('%d_chr_in_cr.png' % c), x_chr[0,:,:,0])
                        cv2.imwrite(CHKDIR + ('%d_chr_in_cb.png' % c), x_chr[0,:,:,1])
                        cv2.imwrite(CHKDIR + ('%d_chr_out.png' % c), y_chr[0,:,:,:])
                    c += 1
    print('%d patches saved.' % c)

if __name__ == '__main__':
    generate_hdf5()
