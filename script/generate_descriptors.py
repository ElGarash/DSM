import os
import pickle

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from cir_net_FOV import *
from distance import *
from OriNet_CVACT.input_data_act_polar_3 import InputData
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import *
import scipy.io as scio

tf.disable_eager_execution()

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='VGG_13_conv_v2_cir')

parser.add_argument('--polar', type=int, help='0 or 1', default=1)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=360)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=360)

args = parser.parse_args()


# --------------  configuration parameters  -------------- #
network_type = args.network_type

polar = args.polar

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

data_type = 'CVACT'

batch_size = 32

DESCRIPTORS_DIRECTORY = '/kaggle/working/descriptors/DSM/'


# -------------------------------------------------------- #

if __name__ == '__main__':
    if os.path.exists(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl") or os.path.exists(f"{DESCRIPTORS_DIRECTORY}/ground_descriptors.pkl"):
        print("Either satellite or ground descriptors already exist on the file system.")
        exit(0)
        
    tf.reset_default_graph()

    # import data
    input_data = InputData(polar)

    # define placeholders
    width = int(test_grd_FOV / 360 * 512)
    grd_x = tf.placeholder(tf.float32, [None, 128, width, 3], name='grd_x')
    polar_sat_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')

    keep_prob = tf.placeholder(tf.float32)

    # build model
    sat_matrix, grd_matrix, distance, pred_orien = VGG_13_conv_v2_cir(polar_sat_x, grd_x, keep_prob, trainable=False)

    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    global_vars = tf.global_variables()

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')

        load_model_path = '/kaggle/working/models/DSM/Model/polar_' + str(polar) + '/' + data_type + '/' + network_type \
                          + '/train_grd_noise_' + str(train_grd_noise) + '/train_grd_FOV_' + str(train_grd_FOV) \
                          + '/model.ckpt'
        saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # ---------------------- validation ----------------------
        print('validate...')
        print('compute global descriptors')
        input_data.reset_scan()
        np.random.seed(2019)

        val_i = 0
        while True:
            print('      progress %d' % val_i)

            batch_sat_polar, batch_grd = input_data.next_batch_scan(batch_size, grd_noise=test_grd_noise, FOV=test_grd_FOV)
            if batch_sat_polar is None:
                break

            feed_dict = {grd_x: batch_grd, polar_sat_x: batch_sat_polar, keep_prob: 1.0}
            sat_matrix_val, grd_matrix_val = sess.run([sat_matrix, grd_matrix], feed_dict=feed_dict)

            sat_global_matrix[val_i: val_i + sat_matrix_val.shape[0], :] = sat_matrix_val
            grd_global_matrix[val_i: val_i + grd_matrix_val.shape[0], :] = grd_matrix_val
            val_i += sat_matrix_val.shape[0]


        if not os.path.exists(DESCRIPTORS_DIRECTORY):
            os.makedirs(DESCRIPTORS_DIRECTORY)

        grd_descriptor = grd_global_matrix
        sat_descriptor = sat_global_matrix

        data_amount = grd_descriptor.shape[0]

        if test_grd_noise==0:
            sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
            sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
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


        with open(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl", 'wb') as f:
            pickle.dump(sat_descriptor, f)


        with open(f"{DESCRIPTORS_DIRECTORY}/ground_descriptors.pkl", 'wb') as f:
            pickle.dump(grd_descriptor, f)

        with open(f"{DESCRIPTORS_DIRECTORY}/dist_array_total.pkl", 'wb') as f:
            pickle.dump(dist_array, f)
