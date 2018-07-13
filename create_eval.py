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
    SRCDIR = '../Data/stereoSR_mb/eval/'
    PARALLAX = 64
    SCALE = 2
    H = 240
    W = 320

    fpdata_eval   = sorted(glob.glob(SRCDIR + '*.png'))
    numPics_eval   = len(fpdata_eval) / 2

    print('%d eval images' % numPics_eval)

    DSTDIR = './data/eval/'

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    
    for i in xrange(numPics_eval):
        print("\t eval image [%2d/%2d]" % (i+1, numPics_eval))
        sys.stdout.flush()

        fleft  = SRCDIR + 'im%d_0.png' % (i)
        fright = SRCDIR + 'im%d_1.png' % (i)
        imgL  = cv2.imread(fleft , flags=cv2.IMREAD_COLOR)
        imgR  = cv2.imread(fright, flags=cv2.IMREAD_COLOR)

        #imgL_rs = cv2.resize(imgL, (int(imgL.shape[1]/2), int(imgL.shape[0]/2)))
        #imgR_rs = cv2.resize(imgR, (int(imgR.shape[1]/2), int(imgR.shape[0]/2)))
        imgL_rs = cv2.resize(imgL, (W,H))
        imgR_rs = cv2.resize(imgR, (W,H))

        imgL_s = cv2.resize(imgL_rs, (int(imgL_rs.shape[1]/SCALE), int(imgL_rs.shape[0]/SCALE)))
        imgR_s = cv2.resize(imgR_rs, (int(imgR_rs.shape[1]/SCALE), int(imgR_rs.shape[0]/SCALE)))

        imgL_ss = cv2.resize(imgL_s, (int(imgL_s.shape[1]*SCALE), int(imgL_s.shape[0]*SCALE)))
        imgR_ss = cv2.resize(imgR_s, (int(imgR_s.shape[1]*SCALE), int(imgR_s.shape[0]*SCALE)))

        cv2.imwrite(DSTDIR + ('X%d_0.png' % i), imgL_ss)
        cv2.imwrite(DSTDIR + ('X%d_1.png' % i), imgR_ss)
        cv2.imwrite(DSTDIR + ('Y%d_0.png' % i), imgL_rs)

if __name__ == '__main__':
    generate_hdf5()
