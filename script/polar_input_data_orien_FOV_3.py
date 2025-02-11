
import cv2
import random
import numpy as np
from os import path, listdir

class InputData:
    def __init__(self):
        self.__cur_test_id = 0  # for training

        """Return dictionary of aerials path key and tuple of ground dir path, number of taken aerials and number of taken grounds"""
        grds_root_path = "/kaggle/input"
        aerials_root_path = "/kaggle/input/polar-aerial-images/polar_aerial_images/"

        aerial_dirs = listdir(aerials_root_path)
        grd_parts = [f'frame-extraction-{i}-{i+1}' for i in range(0, 19, 2)]
        aerial_files_path = []
        ground_files_path = []

        for grd_part in grd_parts:
            part_path = f'{grds_root_path}/{grd_part}/frames'
            for simple_dir in listdir(part_path): 

                if simple_dir in aerial_dirs:
                    aerial_path = path.join(aerials_root_path, simple_dir)
                    grd_path = path.join(part_path, simple_dir)
                    aerial_dir = sorted(listdir(aerial_path))
                    ground_dir = sorted(listdir(grd_path))
                    num_ground = len(ground_dir)
                    num_aerial = len(aerial_dir)

                    for i in range(num_aerial):
                        aerial_file_path = path.join(aerial_path, aerial_dir[i])
                        grd_file_path = [path.join(grd_path, ground_dir[j]) for j in range(i*5, min(i*5+5, num_ground))]
                        aerial_files_path.append(aerial_file_path)
                        ground_files_path.append(grd_file_path)

        i = 0
        while i < len(ground_files_path):
            if len(ground_files_path[i]) < 1:
                del ground_files_path[i]
                del aerial_files_path[i]
            else:
                i +=1


        ground_files_path = [path[0] for path in ground_files_path] # Retrieve the first ground image and drop the remainingf
        aerial_files_path_normal = [path.replace('/polar-aerial-images/polar_aerial_images/', '/aerial-tiles-extraction-0-5000/aerials/') for path in aerial_files_path]

        SELECTED_INDICES_1 =  [210, 211, 212, 290, 305, 319, 330, 355, 400, 505, 740, 800, 840, 900, 870, 935, 960, 965, 967, 990, 1006, 1020, 1095, 1135, 1200, 1204, 1218, 1229, 1297, 1305, 1311, 1355, 1380, 1382, 1497, 1500, 1585, 1595, 1600, 1900, 1960, 1980, 1995, 2020, 2050, 2210, 2220, 2225, 2280, 2400, 2395, 2437, 2545, 2705, 3010, 3025, 3080, 3110, 3235, 3505, 3870, 4400, 4410, 5500, 6010]
        SELECTED_INDICES_2 =  [7536, 7566, 7596, 7616, 7621, 7626, 7641, 7651, 7661, 7676, 7691, 7706, 7711, 7726, 7741, 7751, 7766, 7796, 7816, 7836, 7846, 7906, 7956, 7996, 8156, 8166, 8231, 8321, 8351, 8481, 8596, 8626, 8646, 8691, 8761, 8781, 8836, 8876, 8896, 8916, 8951, 9011, 9121, 9201, 9336, 9346, 9381, 9571, 9586, 9596, 9601, 9666, 9736, 10166, 12461]
        SELECTED_INDICES = SELECTED_INDICES_1 + SELECTED_INDICES_2
        SELECTED_AERIAL_POLAR = [aerial_files_path[index] for index in SELECTED_INDICES]
        SELECTED_AERIAL_NORMAL = [aerial_files_path_normal[index] for index in SELECTED_INDICES]
        SELECTED_GROUND = [ground_files_path[index] for index in SELECTED_INDICES]

        # self.id_test_list = (aerial_files_path, aerial_files_path_normal, ground_files_path)
        self.id_test_list = (SELECTED_AERIAL_POLAR, SELECTED_AERIAL_NORMAL, SELECTED_GROUND)
        self.test_data_size = len(self.id_test_list[0])

        print(f'Number of polar aerial images {len(self.id_test_list[0])}')
        print(f'Number of aerial images {len(self.id_test_list[1])}')
        print(f'Number of ground images {len(self.id_test_list[2])}')


    def next_batch_scan(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None, None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        grd_width = int(FOV/360*512)

        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype = np.float32)
        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        grd_shift = np.zeros([batch_size], dtype=np.int)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
#            print(self.id_test_list[img_idx][0])
                
            # satellite polar
            img = cv2.imread(self.id_test_list[0][img_idx])
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat_polar[i, :, :, :] = img


            # satellite
            img = cv2.imread(self.id_test_list[1][img_idx])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.id_test_list[2][img_idx])
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            # img -= 100.0
            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img_dup

            grd_shift[i] = random_shift

        self.__cur_test_id += batch_size
#        print(grd_shift[0])

        return batch_sat_polar, batch_sat, batch_grd, (np.around(((512-grd_shift)/512*64)%64)).astype(np.int)



    def next_pair_batch(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None, None, None

        grd_width = int(FOV/360*512)

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        grd_shift = np.zeros([batch_size,], dtype=np.int)
        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # satellite polar
            img = cv2.imread(self.id_list[img_idx][0])
            if img is None or img.shape[0] != 128 or img.shape[1] != 512:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.id_list[img_idx][0], i), img.shape)
                continue
            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6    # Red
            batch_sat_polar[batch_idx, :, :, :] = img

            # satellite
            img = cv2.imread(self.id_list[img_idx][1])
            if img is None or img.shape[0] != 750 or img.shape[1] != 750:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.id_list[img_idx][0], i),
                      img.shape)
                continue
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img

            # ground
            img = cv2.imread(self.id_list[img_idx][2])
            if img is None or img.shape[0] != 224 or img.shape[1] != 1232:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.id_list[img_idx][1], i), img.shape)
                continue
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            img_dup = img[:, ((j-random_shift)%512)[:grd_width], :]

            # img -= 100.0
            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img_dup
            grd_shift[batch_idx] = random_shift

            batch_idx += 1

        self.__cur_id += i

        return batch_sat_polar, batch_sat, batch_grd, (np.around(((512-grd_shift)/512*64)%64)).astype(np.int)


    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0
