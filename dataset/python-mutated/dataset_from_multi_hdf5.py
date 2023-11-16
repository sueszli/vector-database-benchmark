from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb

class HDFSequence(lgb.Sequence):

    def __init__(self, hdf_dataset, batch_size):
        if False:
            print('Hello World!')
        '\n        Construct a sequence object from HDF5 with required interface.\n\n        Parameters\n        ----------\n        hdf_dataset : h5py.Dataset\n            Dataset in HDF5 file.\n        batch_size : int\n            Size of a batch. When reading data to construct lightgbm Dataset, each read reads batch_size rows.\n        '
        self.data = hdf_dataset
        self.batch_size = batch_size

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        return self.data[idx]

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.data)

def create_dataset_from_multiple_hdf(input_flist, batch_size):
    if False:
        print('Hello World!')
    data = []
    ylist = []
    for f in input_flist:
        f = h5py.File(f, 'r')
        data.append(HDFSequence(f['X'], batch_size))
        ylist.append(f['Y'][:])
    params = {'bin_construct_sample_cnt': 200000, 'max_bin': 255}
    y = np.concatenate(ylist)
    dataset = lgb.Dataset(data, label=y, params=params)
    dataset.save_binary('regression.train.from_hdf.bin')

def save2hdf(input_data, fname, batch_size):
    if False:
        return 10
    'Store numpy array to HDF5 file.\n\n    Please note chunk size settings in the implementation for I/O performance optimization.\n    '
    with h5py.File(fname, 'w') as f:
        for (name, data) in input_data.items():
            (nrow, ncol) = data.shape
            if ncol == 1:
                chunk = (nrow,)
                data = data.values.flatten()
            else:
                chunk = (batch_size, ncol)
            f.create_dataset(name, data=data, chunks=chunk, compression='lzf')

def generate_hdf(input_fname, output_basename, batch_size):
    if False:
        print('Hello World!')
    df = pd.read_csv(input_fname, header=None, sep='\t')
    mid = len(df) // 2
    df1 = df.iloc[:mid]
    df2 = df.iloc[mid:]
    fname1 = f'{output_basename}1.h5'
    fname2 = f'{output_basename}2.h5'
    save2hdf({'Y': df1.iloc[:, :1], 'X': df1.iloc[:, 1:]}, fname1, batch_size)
    save2hdf({'Y': df2.iloc[:, :1], 'X': df2.iloc[:, 1:]}, fname2, batch_size)
    return [fname1, fname2]

def main():
    if False:
        for i in range(10):
            print('nop')
    batch_size = 64
    output_basename = 'regression'
    hdf_files = generate_hdf(str(Path(__file__).absolute().parents[1] / 'regression' / 'regression.train'), output_basename, batch_size)
    create_dataset_from_multiple_hdf(hdf_files, batch_size=batch_size)
if __name__ == '__main__':
    main()