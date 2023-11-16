import os
import pickle
import sys
import time
import numpy as np
import scipy.sparse
from sklearn.datasets import load_svmlight_file
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter, _utils
from . import constants
from . import syssettings
torch.set_default_tensor_type(syssettings.torch.tensortype)
sparsetensor = syssettings.torch.sparse.tensortype
BYTESPERREAL = 8.0
BYTESPERGB = 1024.0 ** 3

class PrepareData(Dataset):

    def __init__(self, path_data=None, data_format=constants.DataFormat.NUMPY, D=None, N=None, classification=True, ordinal=False, balanced=True, preprocess=None, n_to_estimate=None, MAXMEMGB=syssettings.MAXMEMGB, set_params=True, path_mappings=None, X=None, y=None, verbose=0, n_classes=None, device=constants.Device.CPU):
        if False:
            return 10
        '\n        Dataset class with helpful features and functions for being included in a dataloader\n        and managing memory usage.\n        can read following formats:\n            svm:        svm light format (sklearn.datasets.load_svmlight_file)\n            numpy:      Pass X and y as numpy or sparse arrays\n\n        assumes\n            1. if classification, y is in {-1, 1} or continuous and 0 indexed\n            2. y can fit into memory\n            3. consecutive calls to __getitem__() have consecutive idx values\n\n        notes:\n            1. this implementation is not careful wrt/ precise memory reqts. for\n            example, being able to store one dense row in memory is necessary,\n            but not sufficient.\n            2. for y with 4.2 billion elements, 31.3 GB of memory is  necessary\n            @ 8 bytes/scalar. Use partial fit to avoid loading the entire dataset\n            at once\n            3. disk_size always refer to size of complete data file, even after\n            a split().\n\n\n        Parameters\n        ----------\n        path_data : str\n            Path to load data from\n        data_format : str\n            File ending for path data.\n            "numpy" is the default when passing in X and y\n        D : int\n            Number of features.\n        N : int\n            Number of rows.\n        classification : bool\n            If True, problem is classification, else regression.\n        ordinal: bool\n            If True, problem is ordinal classification. Requires classification to be True.\n        balanced : bool\n            If true, each class is weighted equally in optimization, otherwise\n            weighted is done via support of each class. Requires classification to be True.\n        prerocess : str\n            \'zscore\' which refers to centering and normalizing data to unit variance or\n            \'center\' which only centers the data to 0 mean\n        n_to_estimate : int\n            Number of rows of data to estimate\n        MAXMEMGB : float\n            Maximum allowable size for a minibatch\n        set_params : bool\n            Whether or not to determine the statistics of the dataset\n        path_mappings : str\n            Used when streaming from disk\n        X : array-like\n            Shape = [n_samples, n_features]\n            The training input samples.\n        y : array-like\n            Shape = [n_samples]\n            The target values (class labels in classification, real numbers in\n            regression).\n        verbose : int\n            Controls the verbosity when fitting. Set to 0 for no printing\n            1 or higher for printing every verbose number of gradient steps.\n        device : str\n            \'cpu\' to run on CPU and \'cuda\' to run on GPU. Runs much faster on GPU\n        n_classes : int\n            number of classes\n        '
        self.path_data = path_data
        if self.path_data:
            self.disk_size = os.path.getsize(path_data)
        else:
            assert X is not None, 'X must be specified if no path data'
            self.disk_size = X.nbytes if not scipy.sparse.issparse(X) else X.data.nbytes
        assert data_format in constants.DataFormat.ALL_FORMATS, 'Format must in {0}.'.format(', '.join(constants.DataFormat.ALL_FORMATS))
        self.format = data_format
        self.classification = classification
        self.ordinal = ordinal
        self.balanced = balanced
        self.MAXMEMGB = MAXMEMGB
        self.preprocess = preprocess
        self.set_params = set_params
        self.verbose = verbose
        self.n_classes = n_classes
        self.device = device
        self.path_data_stats = None
        if D is None:
            assert self.disk_size / BYTESPERGB <= self.MAXMEMGB, 'Cannot load data into memory. Supply D.'
            if self.format == constants.DataFormat.SVM:
                (self.X, self.y) = load_svmlight_file(path_data)
            elif self.format == constants.DataFormat.NUMPY:
                assert X is not None, 'X must be specified in numpy mode'
                assert y is not None, 'y must be specified in numpy mode'
                self.X = X
                self.y = y
                if self.n_classes is None:
                    self.n_classes = np.unique(y).shape[0]
                elif self.classification:
                    assert self.n_classes >= np.unique(y).shape[0], 'n_classes given must be greater than or equal to the number of classes in y'
            else:
                raise NotImplementedError
            self.y = torch.as_tensor(self.y, dtype=torch.get_default_dtype())
            (self.N, self.D) = self.X.shape
            self.storage_level = constants.StorageLevel.SPARSE if scipy.sparse.issparse(self.X) else constants.StorageLevel.DENSE
        else:
            assert N is not None, 'Supply N.'
            (self.N, self.D) = (N, D)
            self.storage_level = constants.StorageLevel.DISK
        self.dense_size_gb = self.get_dense_size()
        self.set_dense_X()
        self.max_rows = int(self.MAXMEMGB * BYTESPERGB / BYTESPERREAL / self.D)
        assert self.max_rows, 'Cannot fit one dense row into %d GB memory.' % self.MAXMEMGB
        self.max_rows = self.max_batch_size()
        sys.stdout.flush()
        if n_to_estimate is None:
            self.n_to_estimate = self.max_batch_size()
        else:
            assert n_to_estimate <= self.N, 'n_to_estimate must be <= N.'
            self.n_to_estimate = n_to_estimate
        if self.storage_level == constants.StorageLevel.DISK and self.set_params:
            if self.format == constants.DataFormat.SVM:
                raise NotImplementedError('Please use partial fit to train on datasets that do not fit in memory')
            else:
                raise NotImplementedError
        self.ix_statistics = np.random.permutation(self.N)[:self.n_to_estimate]
        self.n_features = self.D
        if self.set_params:
            if self.verbose:
                print('Finding data statistics...', end='')
                sys.stdout.flush()
            (Xmn, sv1, Xsd, ymn, ysd) = self.compute_data_stats()
            self.set_data_stats(Xmn, sv1, Xsd, ymn, ysd)
            if self.verbose:
                print()
            self.set_return_raw(False)
        else:
            self.set_return_raw(True)
        self.set_return_np(False)
        if self.storage_level == constants.StorageLevel.DISK and self.format == constants.DataFormat.SVM and self.set_params:
            self.loader.batchsize = 1

    def get_dense_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self.N * self.D * BYTESPERREAL / BYTESPERGB

    def set_dense_X(self):
        if False:
            print('Hello World!')
        if self.storage_level != constants.StorageLevel.DISK:
            if self.dense_size_gb <= self.MAXMEMGB:
                if self.storage_level == constants.StorageLevel.SPARSE:
                    self.X = self.X.toarray()
                self.X = torch.as_tensor(self.X, dtype=torch.get_default_dtype())
                self.storage_level = constants.StorageLevel.DENSE

    def set_return_np(self, boolean):
        if False:
            for i in range(10):
                print('nop')
        self.return_np = boolean

    def set_return_raw(self, boolean):
        if False:
            i = 10
            return i + 15
        self.return_raw = boolean

    def save_data_stats(self, path_data_stats):
        if False:
            while True:
                i = 10
        '\n        Dumps dataset statistics to pickle file.\n        '
        data_stats = {'Xmn': self.Xmn, 'sv1': self.sv1, 'Xsd': self.Xsd, 'ymn': self.ymn, 'ysd': self.ysd, 'ix_statistics': self.ix_statistics}
        pickle.dump(data_stats, open(path_data_stats, 'wb'))

    def load_data_stats(self, path_data_stats):
        if False:
            i = 10
            return i + 15
        stats = pickle.load(open(path_data_stats, 'rb'))
        self.path_data_stats = path_data_stats
        self.set_data_stats(np.asarray(stats['Xmn']), stats['sv1'], stats['Xsd'], stats['ymn'], stats['ysd'])
        if self.storage_level == constants.StorageLevel.DISK and hasattr(self, 'path_mappings'):
            if 'ix_statistics' in stats:
                self.ix_statistics = stats['ix_statistics']
            else:
                self.ix_statistics = range(self.N)
        self.set_return_raw(False)

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Resets the dataloader. Only implemented for disk StorageLevel.\n        '
        if self.storage_level == constants.StorageLevel.DENSE:
            pass
        elif self.storage_level == constants.StorageLevel.SPARSE:
            pass
        elif self.storage_level == constants.StorageLevel.DISK:
            if self.format == constants.DataFormat.SVM:
                self.loader.reset()
            else:
                raise NotImplementedError

    def todense(self):
        if False:
            return 10
        assert hasattr(self, 'Xmn'), 'Set preprocess params first.'
        assert len(self) <= self.max_batch_size(), 'N must be <= max_batch_size().'
        with torch.no_grad():
            (dense, _) = self.split(range(len(self)))
            Braw = self.return_raw
            Bnp = self.return_np
            self.set_return_raw(True)
            self.set_return_np(True)
            (dense.X, dense.y) = ([], [])

            def f_Xy(X, y):
                if False:
                    while True:
                        i = 10
                dense.X.append(X)
                dense.y.append(y)
            self.apply(f_Xy=f_Xy)
            dense.X = dense.X[-1]
            dense.y = dense.y[-1]
            self.set_return_raw(Braw)
            self.set_return_np(Bnp)
            dense.storage_level = constants.StorageLevel.DENSE
            return dense

    def split(self, ix):
        if False:
            for i in range(10):
                print('nop')
        assert hasattr(self, 'Xmn'), 'Run set_preprocess_params() first.'
        first = type(self)(self.path_data, self.format, self.D, N=len(ix), classification=self.classification, preprocess=self.preprocess, n_to_estimate=None, MAXMEMGB=self.MAXMEMGB, set_params=False)
        second = type(self)(self.path_data, self.format, self.D, N=self.N - len(ix), classification=self.classification, preprocess=self.preprocess, n_to_estimate=None, MAXMEMGB=self.MAXMEMGB, set_params=False)
        first.storage_level = self.storage_level
        second.storage_level = self.storage_level
        if not self.classification:
            first.ymn = self.ymn
            second.ymn = self.ymn
            first.ysd = self.ysd
            second.ysd = self.ysd
        first.Xmn = self.Xmn
        second.Xmn = self.Xmn
        first.sv1 = self.sv1
        second.sv1 = self.sv1
        if self.storage_level == constants.StorageLevel.DISK:
            if self.format == constants.DataFormat.SVM:
                first.Xsd = self.Xsd
                second.Xsd = self.Xsd
            else:
                raise NotImplementedError
        if self.storage_level == constants.StorageLevel.DISK:
            if self.format == constants.DataFormat.SVM:
                raise NotImplementedError
            raise NotImplementedError
        elif self.storage_level in [constants.StorageLevel.SPARSE, constants.StorageLevel.DENSE]:
            (first.X, first.y) = (self.X[ix], self.y[ix])
            ixsec = list(set(range(self.N)).difference(set(ix)))
            (second.X, second.y) = (self.X[ixsec], self.y[ixsec])
        return (first, second)

    @staticmethod
    def sparse_std(X, X_mean):
        if False:
            print('Hello World!')
        '\n        Calculate the column wise standard deviations of a sparse matrix.\n        '
        X_copy = X.copy()
        X_copy.data **= 2
        E_x_squared = np.array(X_copy.mean(axis=0)).ravel()
        Xsd = np.sqrt(E_x_squared - X_mean ** 2)
        return Xsd

    def compute_data_stats(self):
        if False:
            print('Hello World!')
        "\n        1. computes/estimates feature means\n        2. if preprocess == 'zscore', computes/estimates feature standard devs\n        3. if not classification, computes/estimates target mean/standard dev\n        4. estimates largest singular value of data matrix\n        "
        t = time.time()
        (X, y) = (self.X[self.ix_statistics], self.y[self.ix_statistics])
        preprocess = self.preprocess
        classification = self.classification
        Xmn = X.mean(dim=0) if not scipy.sparse.issparse(X) else np.array(X.mean(axis=0)).ravel()
        if preprocess == constants.Preprocess.ZSCORE:
            Xsd = X.std(dim=0) if not scipy.sparse.issparse(X) else PrepareData.sparse_std(X, Xmn)
            Xsd[Xsd == 0] = 1.0
        else:
            Xsd = 1.0
        if preprocess is not None and preprocess:
            if preprocess == constants.Preprocess.ZSCORE:
                Xc = (X - Xmn) / Xsd
            else:
                Xc = X - Xmn
        else:
            Xc = X - Xmn
        sv1 = scipy.sparse.linalg.svds(Xc / (torch.sqrt(torch.prod(torch.as_tensor(y.size(), dtype=torch.get_default_dtype()))) if not scipy.sparse.issparse(X) else y.numpy().size), k=1, which='LM', return_singular_vectors=False)
        sv1 = np.array([min(np.finfo(np.float32).max, sv1[0])])
        if not classification:
            ymn = y.mean()
            ysd = y.std()
        else:
            ymn = 0.0
            ysd = 1.0
        if self.verbose:
            print(' computing data statistics took: ', time.time() - t)
        return (Xmn, sv1, Xsd, ymn, ysd)

    def set_data_stats(self, Xmn, sv1, Xsd=1.0, ymn=0.0, ysd=1.0):
        if False:
            print('Hello World!')
        '\n        Saves dataset stats to self to be used for preprocessing.\n        '
        self.Xmn = torch.as_tensor(Xmn, dtype=torch.get_default_dtype()).to(self.device)
        self.sv1 = torch.as_tensor(sv1, dtype=torch.get_default_dtype()).to(self.device)
        self.Xsd = torch.as_tensor(Xsd, dtype=torch.get_default_dtype()).to(self.device)
        self.ymn = torch.as_tensor(ymn, dtype=torch.get_default_dtype()).to(self.device)
        self.ysd = torch.as_tensor(ysd, dtype=torch.get_default_dtype()).to(self.device)

    def apply_preprocess(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        Faster on gpu device, while dataloading takes up a large portion of the time.\n        '
        with torch.no_grad():
            if not self.classification:
                y = (y.reshape((-1, 1)) - self.ymn) / self.ysd
            else:
                y = y.reshape((-1, 1))
            X = (X - self.Xmn) / self.sv1
            if self.preprocess == constants.Preprocess.ZSCORE:
                X /= self.Xsd
            return (X, y)

    def max_batch_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the maximum batchsize for the dataset.\n        '
        return int(np.min([self.max_rows, self.N]))

    def apply(self, ix_rows=None, ix_cols=None, f_Xy=None):
        if False:
            i = 10
            return i + 15
        if f_Xy is None:
            return
        if ix_rows is None:
            ix_rows = range(self.N)
        if ix_cols is None:
            ix_cols = range(self.n_features)
        f_Xy(self.X[ix_rows, ix_cols] if not self.storage_level == constants.StorageLevel.SPARSE else self.X[ix_rows, ix_cols].toarray(), self.y[ix_rows])

    def get_dense_data(self, ix_cols=None, ix_rows=None):
        if False:
            for i in range(10):
                print('nop')
        if ix_cols is None:
            ix_cols = range(self.n_features)
        X = [np.zeros((0, len(ix_cols)))]
        y = [np.zeros((0, 1))]
        Bnp = self.return_np

        def f_Xy(Xb, yb, n):
            if False:
                while True:
                    i = 10
            X[-1] = np.concatenate((X[-1], Xb), axis=0)
            y[-1] = np.concatenate((y[-1], yb), axis=0)
        self.apply(f_Xy=f_Xy, ix_rows=ix_rows, ix_cols=ix_cols)
        self.set_return_np(Bnp)
        return (X[-1], y[-1])

    def __len__(self):
        if False:
            print('Hello World!')
        return self.N

    def getXy(self, idx):
        if False:
            print('Hello World!')
        if self.storage_level == constants.StorageLevel.DENSE:
            (X, y) = (self.X[idx], self.y[idx])
        elif self.storage_level == constants.StorageLevel.SPARSE:
            (X, y) = (self.X[idx].toarray(), self.y[idx])
        else:
            raise NotImplementedError
        return (X, y)

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            (X, y) = self.getXy(idx)
            X = X.toarray() if scipy.sparse.issparse(X) else X
            X = torch.as_tensor(X, dtype=torch.get_default_dtype()).to(self.device)
            y = torch.as_tensor(y, dtype=torch.get_default_dtype()).to(self.device)
            if not self.return_raw:
                (X, y) = self.apply_preprocess(X, y)
            if self.classification and (self.n_classes is None or self.n_classes == 2):
                y[y == 0] = -1
            if self.return_np:
                if constants.Device.CPU not in self.device:
                    X = X.cpu()
                    y = y.cpu()
                X = X.numpy()
                y = y.numpy()
                return (X, y)
            return (X, y)

class ChunkDataLoader(DataLoader):
    """
    DataLoader class used to more quickly load a batch of indices at once.
    """

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return _ChunkDataLoaderIter(self)

class _ChunkDataLoaderIter:
    """
    DataLoaderIter class used to more quickly load a batch of indices at once.
    """

    def __init__(self, dataloader):
        if False:
            print('Hello World!')
        if dataloader.num_workers == 0:
            self.iter = _SingleProcessDataLoaderIter(dataloader)
        else:
            self.iter = _MultiProcessingDataLoaderIter(dataloader)

    def __next__(self):
        if False:
            while True:
                i = 10
        if self.iter._num_workers == 0:
            indices = next(self.iter._sampler_iter)
            if len(indices) > 1:
                batch = self.iter._dataset[np.array(indices)]
            else:
                batch = self.iter._collate_fn([self.iter._dataset[i] for i in indices])
            if self.iter._pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            return batch
        else:
            return next(self.iter)