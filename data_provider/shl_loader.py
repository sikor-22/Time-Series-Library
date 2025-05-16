from torch.utils.data import Dataset
import posixpath as path
import numpy as np
import pandas as pd


class SHLloader(Dataset):
    '''
    Dataset for SHL Locomotion Challenge. Please note that it expects the following folder structure
    from root_path.
    Additional args:
        args.sensor_location: one of Hips/Torso/Bag/Hand
        args.sensor: one of Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z, Mag_x, Mag_y, Mag_z
    Data Folder structure:
        - root_path
            - train
                - Bag
                - Hand
                - Torso
                - Hips
            - validation
                - Bag
                - Hand
                - Torso
                - Hips
            - test
    '''
    def __init__(self, args, root_path, flag="train"):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        if flag == "TEST":
            self.flag = "validation"
        if isinstance(self.flag, str):
            self.flag = self.flag.lower() # Avoid ambiguities with Train/TRAIN/train etc
        self.x_data, self.y_data = self.__load_from_file()
        self.max_seq_len = 500
        self.feature_df = np.zeros_like(self.x_data.T)
        self.class_names = np.unique(self.y_data).astype(str)

    def __load_from_file(self, sensor = None, location = None):
        '''
        Load singular data file decided by self.args
        Args:
            - sensor: which modality to load (see __init__), if None args.sensor will be used
            - location: which location to load (see __init__), if None args.sensor_location will be used
        '''
        if sensor is None:
            sensor = self.args.sensor
        if location is None:
            location = self.args.sensor_location
        if self.flag == 'test':
            filepath = path.join(self.root_path, self.flag, f"{sensor}.txt")
        else:
            filepath = path.join(self.root_path, self.flag, location, f"{sensor}.txt")
        np_data = np.loadtxt(filepath, dtype=np.float32)
        np_data = self._preprocess_data(np_data)
        self.x = np_data
        labelpath = path.join(self.root_path, self.flag, location, "Label.txt")
        y_data = np.loadtxt(labelpath, dtype=int)
        y_data = np.median(y_data, axis=1).astype(int)
        return np_data, y_data


    def _preprocess_data(self, data):
        '''
        Placeholder function for further development
        '''
        return data

    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, ind):
        return self.x_data[ind], self.y_data[ind]