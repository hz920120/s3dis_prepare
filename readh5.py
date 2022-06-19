import os

import numpy as np
from h5py import Dataset
import h5py
import glob
import tqdm


class S3DISDataSetH5(Dataset):
    def __init__(self, root="./data/S3DIS_hdf5", split="train",
                 num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        '''
            root: Data set path
            split: Training set or test set("train","test")
            num_point: Sampling points
            test_area: Test set Area_5. Other numbers can also be taken. In the paper, 5 is taken
            block_size: Change the sampling room to block_size * block_size Size of, unit: m
            sample_rate: Sampling rate, 1 indicates full sampling
            transform: I don't know at present. Follow up
        '''

        self.num_point = num_point     # Sampling points
        self.block_size = block_size   # Change the sampling room to block_size * block_size, unit: m
        self.transform = transform
        self.room_points, self.room_labels = [], []   # Point cloud data, label value (refers to: in a point cloud file, label is added to each line of data)
        self.room_coord_min, self.room_coord_max = [], []    # Minimum value and maximum value of each dimension (X, Y, Z) of each room (point cloud file)

        num_point_all = []     # Total number of points in each room
        labelweights = np.zeros(13)       # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        rooms = [os.path.basename(file) for file in glob.glob(root+"/*.h5")]     # Dataset file, ['area_1_conferenceroom_1. H5 ','area_1_conferenceroom_2. H5',...,'area_6_pantry_1. H5 ']
        rooms = [room for room in rooms if "Area_" in room]

        # Data set segmentation
        # room.split("_")[1] : Area_1_WC_1.h5'.split("_")[1]
        if split=="train":
            rooms_split = [room for room in rooms if int(room.split("_")[1]) != test_area]
        else:
            rooms_split = [room for room in rooms if int(room.split("_")[1]) == test_area]


        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(root, room_name)     # Get dataset file/ data/S3DIS_hdf5\Area_1_WC_1.h5

            # Read h5 file
            f = h5py.File(room_path)
            points = np.array(f["data"])     # [N, 6]  XYZRGB
            labels = np.array(f["label"])    # [N,]    L

            f.close()

            tmp,_ = np.histogram(labels, range(14))
            labelweights = labelweights + tmp       # Count the number of point categories in all room s

            coord_min, coord_max = np.min(points, 0)[:3], np.max(points, 0)[:3]

            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)     # Proportion of all kinds of points in the total points
        # Maximum value / labelweights, function: give more weight to those with the least number of categories
        # Third power: in order to flatten the weights, they are not easy to change
        self.labelweights = np.power(np.max(labelweights)/labelweights, 1/3.0)

        sample_prob = num_point_all / np.sum(num_point_all)      # Proportion of point cloud number of each room to total point cloud number
        num_iter = int( sample_rate * np.sum(num_point_all) / num_point )     # sample_rate * total points / sampling points of all rooms. A total of num iterations are required_ ITER times to sample all rooms
        room_idxs = []
        for index in range(len(rooms_split)):
            # sample_prob[index]: the proportion of the number of point clouds corresponding to room in the total number of point clouds; num_iter: total iterations
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))       # sample_prob[index] * num_iter: the number of times needed to sample the index room
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))   # len(self.room_idxs): 47576
        # len(room_idxs) == num_iter

    def __getitem__(self, index):
        room_idx = self.room_idxs[index]
        points = self.room_points[room_idx]   # N × 6
        labels = self.room_labels[room_idx]   # N × 1

        N = points.shape[0]   # Number of points

        while(True):
            center = points[np.random.choice(N), :3]    # Randomly assign a point as the center of the block
            # 1m  ×  1m range
            block_min = center - [self.block_size/2.0, self.block_size/2.0, 0]
            block_max = center + [self.block_size/2.0, self.block_size/2.0, 0]
            '''
                np.where(condition, a, b): satisfy condition，fill a，Otherwise fill b
                    If not a,b，only np.where(condition)，Returns:(array1, array2)，array1 Rows that meet the criteria, array2: Qualified columns 
            '''
            # The index of the selected point within the range of the block
            point_index = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_index.shape[0]>1024:
                break

        # sampling
        if point_index.shape[0] >= self.num_point:
            sample_point_index = np.random.choice(point_index, self.num_point, replace=False)
        else:
            sample_point_index = np.random.choice(point_index, self.num_point, replace=True)

        sample_points = points[sample_point_index, :]    # num_point × 6

        # normalization
        current_points = np.zeros([self.num_point, 9])   # num_point  ×  9. XYZRGBX'Y'Z ', X': coordinates after X normalization
        current_points[:, 6] = sample_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = sample_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = sample_points[:, 2] / self.room_coord_max[room_idx][2]
        sample_points[:, 0] = sample_points[:, 0] - center[0]
        sample_points[:, 1] = sample_points[:, 1] - center[1]
        sample_points[:, 3:6] = sample_points[:, 3:6] / 255
        current_points[:, 0:6] = sample_points
        current_labels = labels[sample_point_index]

        if self.transform:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels


    def __len__(self):
        return len(self.room_idxs)