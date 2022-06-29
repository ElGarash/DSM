import cv2
import random
import numpy as np
# load the yaw, pitch angles for the street-view images and yaw angles for the aerial view
import scipy.io as sio


class InputData:
    # the path of your CVACT dataset

    img_root = '/kaggle/input/cvact-small/'
    img_root_polar = '/kaggle/input/cvact-polar-images-dsm/cvact-small/'

    # yaw_pitch_grd = sio.loadmat('./CVACT_orientations/yaw_pitch_grd_CVACT.mat')
    # yaw_sat = sio.loadmat('./CVACT_orientations/yaw_radius_sat_CVACT.mat')

    posDistThr = 25
    posDistSqThr = posDistThr * posDistThr

    panoCropPixels = int(832 / 2)

    panoRows = 128

    panoCols = 512

    satSize = 256

    def __init__(self, polar=1):
        self.polar = polar

        self.allDataList = '/kaggle/input/cvact-small/ACT_data.mat'
        print('InputData::__init__: load %s' % self.allDataList)

        self.id_alllist = []
        self.id_idx_alllist = []

        # load the mat

        anuData = sio.loadmat(self.allDataList)

        idx = 0
        for i in range(0, len(anuData['panoIds'])):
            grd_id_align = self.img_root_polar + 'streetview_polish/' + anuData['panoIds'][i] + '_grdView.jpg'
            polar_sat_id_ori = self.img_root_polar + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.jpg'

            self.id_alllist.append([grd_id_align, polar_sat_id_ori])
            self.id_idx_alllist.append(idx)
            idx += 1
        self.all_data_size = len(self.id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', self.all_data_size)


        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        # self.valNum = len(self.val_inds)
        self.valNum = 100

        self.valList = []
        for k in range(self.valNum):
            self.valList.append(self.id_alllist[self.val_inds[k][0]])
        # cur validation index
        self.__cur_test_id = 0

    def next_batch_scan(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_test_id >= self.valNum:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.valNum:
            batch_size = self.valNum - self.__cur_test_id

        batch_polar_sat = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)
        grd_width = int(FOV / 360 * 512)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # polar satellite
            img = cv2.imread(self.valList[img_idx][-1])
            # img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)


            if img is None or img.shape[0] != self.panoRows or img.shape[1] != self.panoCols:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][-1], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_polar_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.valList[img_idx][0])
            # img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)


            if img is None or img.shape[0] * 4 != img.shape[1]:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][0], i))
                continue

            img = img.astype(np.float32)

            j = np.arange(0, 512)
            a = np.random.rand()
            random_shift = int(a * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img_dup



        self.__cur_test_id += batch_size


        return batch_polar_sat, batch_grd

    #
    def get_test_dataset_size(self):
        return self.valNum

    #
    def reset_scan(self):
        self.__cur_test_id = 0


if __name__ == '__main__':
    input_data = InputData()
    batch_sat, batch_grd = input_data.next_batch_scan(12)