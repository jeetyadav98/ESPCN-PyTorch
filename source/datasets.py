import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        """ Constructor

        Simple class to store the train dataset (.h5 file), for usage in training. Other member functions include standard __getitem__ and a return-length.

        :param h5_file: path string to the train .h5 file
        :return: None

        """
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        """ Standard getitem function for loading datasets
        """
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        """ Length
        """
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        """ Constructor

        Simple class to store the train dataset (.h5 file), for usage in training. Other member functions include standard __getitem__ and a return-length.

        :param h5_file: path string to the train .h5 file
        :return: None

        """
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        """ Standard getitem function for loading datasets
        """
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        """ Length
        """
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
