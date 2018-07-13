import time

from utils import *
import random

def stereoSR(x_lum0, x_lum, x_chr):
    res = tf.concat([x_lum0, x_lum], 3)
    tf.summary.scalar('x_lum0_hat_min_0', tf.reduce_min(x_lum0))
    tf.summary.scalar('x_lum0_hat_max_0', tf.reduce_max(x_lum0))
    tf.summary.scalar('x_lum0_hat_mean_0', tf.reduce_mean(x_lum0))
    tf.summary.scalar('res_hat_min_0', tf.reduce_min(res))
    tf.summary.scalar('res_hat_max_0', tf.reduce_max(res))
    tf.summary.scalar('res_hat_mean_0', tf.reduce_mean(res))
    for i in range(15):
        res = tf.layers.conv2d(res, 64, 3, padding='same', activation=tf.nn.relu, name='conv%d' % (i+1))
        tf.summary.scalar('res_hat_min_%d' % (i+1), tf.reduce_min(res))
        tf.summary.scalar('res_hat_max_%d' % (i+1), tf.reduce_max(res))
        tf.summary.scalar('res_hat_mean_%d' % (i+1), tf.reduce_mean(res))
    res = tf.layers.conv2d(res, 1, 3, padding='same', name='conv16', activation=None)
    tf.summary.scalar('res_hat_min_16', tf.reduce_min(res))
    tf.summary.scalar('res_hat_max_16', tf.reduce_max(res))
    tf.summary.scalar('res_hat_mean_16', tf.reduce_mean(res))

    hr_lum = tf.add(res, x_lum0)
    ycrcb = tf.concat([hr_lum, x_chr], 3)  ###!!! check the order
    
    return ycrcb, res

class imdualenh(object):
    def __init__(self, sess, batch_size=128, PARALLAX=64):
        self.sess = sess
        self.parallax = PARALLAX
        
        # build model

        # Labels
        self.Y_     = tf.placeholder(tf.float32, [None, None, None, 3], name='hr_gt_ycrcb_p') # HR GT YCrCb
        self.R_     = tf.placeholder(tf.float32, [None, None, None, 1], name='res_gt_p') # RES GT LUM
        
        # Inputs
        self.X_lum0 = tf.placeholder(tf.float32, [None, None, None, 1], name='lr_lum_p') # LR LUM Left
        self.X_lum  = tf.placeholder(tf.float32, [None, None, None, self.parallax], name='lr_lum_shf_p') # LR LUM Shifted Right (64)
        self.X_chr  = tf.placeholder(tf.float32, [None, None, None, 2], name='lr_chr_p') # LR CHR Left

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.Y, self.R = stereoSR(self.X_lum0, self.X_lum, self.X_chr)

        tf.summary.image('RES_GT_LUM'   , self.R_, 2) # GT: ground truth
        tf.summary.image('RES_HAT_LUM'  , self.R , 2) # HAT: estimated
        tf.summary.image('HR_HAT_YCrCb' , self.Y , 2) # LUM: luminance
        tf.summary.image('HR_GT_YCrCb'  , self.Y_, 2) # LUM: luminance
        tf.summary.image('LR_LUM', self.X_lum0   , 2) # HR, LR: high-res., low-res.
        tf.summary.scalar('res_gt_min', tf.reduce_min(self.R_))
        tf.summary.scalar('res_gt_max', tf.reduce_max(self.R_))
        tf.summary.scalar('res_gt_mean', tf.reduce_mean(self.R_))
        
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.R_ - self.R)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
        sys.stdout.flush()

    def evaluate(self, iter_num, test_data_YL, test_data_XL, test_data_XR,
                 sample_dir, summary_merged, summary_writer):
        print("[*] Evaluating...")
        sys.stdout.flush()
        
        psnr_sum = 0
        for idx in xrange(len(test_data_YL)):
            im_h, im_w, ch = test_data_YL[idx].shape
            assert ch == 3
            # inputs
            X_lum0 = np.zeros((1,im_h,im_w-self.parallax,1))
            X_lum0[0,:,:,0] = test_data_XL[idx][:,self.parallax:,0].astype(np.float32) / 255.0
            X_lum  = np.zeros((1,im_h,im_w-self.parallax,self.parallax))
            for p in range(0, self.parallax, 1):
                X_lum[0,:,:,p] = test_data_XR[idx][:,self.parallax-p:im_w-p,0].astype(np.float32) / 255.0
            X_chr = np.zeros((1,im_h, im_w-self.parallax, 2))
            X_chr[0,:,:,:] = test_data_XL[idx][:,self.parallax:,1:].astype(np.float32) / 255.0
            
            # outpus
            Y_GT = np.zeros((1,im_h,im_w-self.parallax,3))
            Y_GT[0,:,:,:] = test_data_YL[idx][:,self.parallax:,:].astype(np.float32) / 255.0

            R_lum_tmp = np.zeros((1,im_h,im_w-self.parallax,1))
            R_lum_tmp[0,:,:,0] = (test_data_YL[idx][:,self.parallax:,0].astype(np.float32) - 
                                  test_data_XL[idx][:,self.parallax:,0].astype(np.float32))
            R_lum_min = R_lum_tmp.min()
            R_lum_max = R_lum_tmp.max()
            R_lum = (R_lum_tmp - R_lum_min) / (R_lum_max - R_lum_min)

            assert R_lum.max() <= 1 and R_lum.min() >= 0

            # run the model
            HR_YCrCb_hat_image, HR_RES_hat_image, psnr_summary = self.sess.run(
                       [self.Y, self.R, summary_merged],
                       feed_dict={self.X_lum0: X_lum0,
                                  self.X_lum: X_lum,
                                  self.X_chr: X_chr,
                                  self.R_: R_lum,
                                  self.Y_: Y_GT,
                                  self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(test_data_YL[idx][:,self.parallax:,:], 0, 255).astype('uint8').squeeze()
            LR_image    = np.clip(test_data_XL[idx][:,self.parallax:,:], 0, 255).astype('uint8').squeeze()
            outputimage = np.clip(255 * HR_YCrCb_hat_image, 0, 255).astype('uint8').squeeze()
            # calculate PSNR
            assert groundtruth.shape[2] == 3
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, LR_image, outputimage, iter_num, idx+1)
        avg_psnr = psnr_sum / len(test_data_YL)
        print("--- Test ---- Average PSNR %.2f ---" % (avg_psnr))
        sys.stdout.flush()

    #def denoise(self, data_gt, data_in):
    #    output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
    #            feed_dict={self.Y_:data_gt, self.X:data_in, self.is_training: False})
    #    return output_clean_image, noisy_image, psnr

    def train(self, data, eval_data_YL, eval_data_XL, eval_data_XR, 
              batch_size, ckpt_dir, epoch, lr, use_gpu, sample_dir,
              eval_every_epoch=2):
        # assert data range is between 0 and 1
        data_num = data["tr_in_lum"].shape[0]
        numBatch = int(data_num / batch_size)
        # load pretrained model
#        load_model_status, global_step = self.load(ckpt_dir)
#        if load_model_status:
#            iter_num = global_step
#            start_epoch = global_step // numBatch
#            start_step = global_step % numBatch
#            print("[*] Model restore success!")
#            sys.stdout.flush()
#        else:
        iter_num = 0
        start_epoch = 0
        start_step = 0
        print("[*] Not find pretrained model!")
        sys.stdout.flush()
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        if use_gpu == 1:
            writer = tf.summary.FileWriter('./logs-gpu', self.sess.graph)
        else:
            writer = tf.summary.FileWriter('./logs-cpu', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        sys.stdout.flush()
        start_time = time.time()
        self.evaluate(iter_num, eval_data_YL, eval_data_XL, eval_data_XR,
                      sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)
        for epoch in xrange(start_epoch, epoch):
            blist = random.sample(range(0, numBatch), numBatch)
            for batch_id in xrange(start_step, numBatch):
                i_s = blist[batch_id] * batch_size
                i_e = min((blist[batch_id] + 1 ) * batch_size, data_num)
                batch_X_lum  = data["tr_in_lum"][i_s:i_e, ..., 1:]
                batch_X_lum0 = np.expand_dims(data["tr_in_lum"][i_s:i_e, ..., 0],3) 
                batch_X_chr  = data["tr_in_chr"][i_s:i_e, ...]

                batch_Y = data["tr_out_chr"][i_s:i_e, ...].astype(np.float32) / 255.0
                batch_lum_tmp = data["tr_out_lum"][i_s:i_e, ...]
                batch_R_tmp = batch_lum_tmp.astype(np.float32) - batch_X_lum0.astype(np.float32)

                batch_R_min = batch_R_tmp.min()
                batch_R_max = batch_R_tmp.max()
                batch_R = (batch_R_tmp - batch_R_min) / (batch_R_max - batch_R_min)

                assert batch_R.max() <= 1 and batch_R.min() >= 0

                batch_X_lum  = batch_X_lum.astype(np.float32) / 255.0
                batch_X_lum0 = batch_X_lum0.astype(np.float32) / 255.0
                batch_X_chr  = batch_X_chr.astype(np.float32) / 255.0

                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                        feed_dict={self.X_lum: batch_X_lum,
                                   self.X_lum0: batch_X_lum0,
                                   self.X_chr: batch_X_chr,
                                   self.R_: batch_R,
                                   self.Y_: batch_Y,
                                   self.lr: lr[epoch], self.is_training: True})
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                sys.stdout.flush()
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data_YL, eval_data_XL, eval_data_XR,
                              sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer)  # eval_data value range is 0-255
                #self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")
        sys.stdout.flush()

    def save(self, iter_num, ckpt_dir, model_name='stereoSR'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        sys.stdout.flush()
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        sys.stdout.flush()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir):
        """Test CNN_PAR"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        sys.stdout.flush()
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        sys.stdout.flush()
        for idx in xrange(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
        sys.stdout.flush()
