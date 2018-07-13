#! /usr/bin/env python2

import argparse
from glob import glob

import tensorflow as tf

from model import imdualenh
from utils import *
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=0, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12', help='dataset for eval in training')
#parser.add_argument('--test_set', dest='test_set', default='BSD68', help='dataset for testing')
parser.add_argument('--hdf5_file', dest='hdf5_file',help='data for training and evaluation.')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, help='evaluation every epoch.')
parser.add_argument('--PARALLAX', dest='PARALLAX', type=int, default=65 ,help='PARALLAX')
args = parser.parse_args()

def model_train(model_name, lr):
    with load_data(filepath=args.hdf5_file) as data:
        eval_files_YL = sorted(glob('{}/Y*_0.png'.format(args.eval_set)))
        eval_files_XL = sorted(glob('{}/X*_0.png'.format(args.eval_set)))
        eval_files_XR = sorted(glob('{}/X*_1.png'.format(args.eval_set)))
        eval_data_YL = load_images(eval_files_YL) # list of array of different size, 4-D, pixel value range is 0-255
        eval_data_XL = load_images(eval_files_XL)
        eval_data_XR = load_images(eval_files_XR) 
        model_name.train(data = data,
                       eval_data_YL=eval_data_YL,
                       eval_data_XL=eval_data_XL,
                       eval_data_XR=eval_data_XR,
                       batch_size=args.batch_size,
                       ckpt_dir=args.ckpt_dir, 
                       sample_dir=args.sample_dir,
                       epoch=args.epoch,
                       lr=lr,
                       use_gpu=args.use_gpu,
                       eval_every_epoch=args.eval_every_epoch)

#def model_test(model_name):
#    test_files = glob('./data/test/{}/*.png'.format(args.test_set))
#    model_name.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0   ###!!!
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        sys.stdout.flush()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = imdualenh(sess, batch_size=args.batch_size, PARALLAX=args.PARALLAX)
            if args.phase == 'train':
                model_train(model, lr=lr)
            elif args.phase == 'test':
                model_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        sys.stdout.flush()
        with tf.Session() as sess:
            model = imdualenh(sess, batch_size=args.batch_size, PARALLAX=args.PARALLAX)
            if args.phase == 'train':
                model_train(model, lr=lr)
            elif args.phase == 'test':
                model_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
