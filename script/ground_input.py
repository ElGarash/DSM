import os
import pickle
import cv2
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from cir_net_FOV import *
from polar_input_data_orien_FOV_3 import InputData
from distance import *

import tensorflow.compat.v1 as tf
import numpy as np

import argparse
from numpy import fft


# Disable eager execution
tf.disable_eager_execution()


parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='VGG_13_conv_v2_cir')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=25)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 100, 120, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 100, 120, 180, 360', default=360)

parser.add_argument("--image_path", type=str, help="The path of the ground image to calculate the distance for")

args = parser.parse_args()
print(args)

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type

start_epoch = args.start_epoch
polar = args.polar

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

data_type = 'CVUSA'

loss_type = 'l1'

batch_size = 32
is_training = False
loss_weight = 10.0


DESCRIPTORS_DIRECTORY = '/kaggle/working/descriptors/DSM'

with open(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl", 'rb') as f:
    sat_descriptor = pickle.load(f)


# -------------------------------------------------------- #

def preprocess_ground_image(image, grd_noise=360, FOV=360):
    grd_width = int(FOV/360*512)

    batch_grd = np.zeros([1, 128, grd_width, 3], dtype = np.float32)

    img = cv2.imread(image)
    img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    j = np.arange(0, 512)
    random_shift = int(np.random.rand() * 512 * grd_noise / 360)
    img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

    # img -= 100.0
    img_dup[:, :, 0] -= 103.939  # Blue
    img_dup[:, :, 1] -= 116.779  # Green
    img_dup[:, :, 2] -= 123.6  # Red
    batch_grd[0, :, :, :] = img_dup
    
    return batch_grd

def compute_loss(dist_array):

    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        pos_dist = tf.diag_part(dist_array)

        pair_n = batch_size * (batch_size - 1.0)

        # satellite to ground
        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

        # ground to satellite
        triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
        loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0

    return loss


if __name__ == '__main__':

    tf.reset_default_graph()

    width = int(test_grd_FOV / 360 * 512)

    # define placeholders
    grd_x = tf.placeholder(tf.float32, [None, 128, width, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')
    polar_sat_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    _, grd_matrix, distance, pred_orien = VGG_13_conv_v2_cir(polar_sat_x, grd_x, keep_prob, is_training)

    loss = compute_loss(distance)

    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    grd_global_matrix = np.zeros([1, g_height, g_width, g_channel])

    print('setting saver...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    print('setting saver done...')

    global_vars = tf.global_variables()

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.global_variables_initializer())

        print('load model...')

        load_model_path = '/kaggle/working/models/DSM/Model/polar_' + str(polar) + '/' + data_type + '/' + network_type \
                          + '/train_grd_noise_' + str(train_grd_noise) + '/train_grd_FOV_' + str(train_grd_FOV) \
                          + '/model.ckpt'
        saver.restore(sess, load_model_path)

        print("Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # ---------------------- validation ----------------------

        np.random.seed(2019)
        batch_grd = preprocess_ground_image(args.image_path, grd_noise=test_grd_noise, FOV=test_grd_FOV)

        feed_dict = {grd_x: batch_grd, keep_prob: 1.0}
        grd_matrix_val = sess.run(grd_matrix, feed_dict=feed_dict)

        grd_global_matrix = grd_matrix_val

        grd_descriptor = grd_global_matrix

        data_amount = 1

        if test_grd_noise==0:
            grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel])
            dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
        else:
            sat_fft = fft.fft(sat_descriptor.transpose([0, 3, 1, 2]))[:, np.newaxis, ...]

            for i in range(100):
                print(i)
                batch_start = int(data_amount * i / 100)
                if i < 99:
                    batch_end = int(data_amount * (i + 1) / 100)
                else:
                    batch_end = data_amount

                dist_array, pred_orien = corr_distance_FOV_np(grd_descriptor[batch_start: batch_end, :], sat_descriptor, sat_fft)


        with open(f"{DESCRIPTORS_DIRECTORY}/dist_array.pkl", 'wb') as f:
            pickle.dump(dist_array, f)