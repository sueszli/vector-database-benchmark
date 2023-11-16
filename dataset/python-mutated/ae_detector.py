from bigdl.chronos.detector.anomaly.abstract import AnomalyDetector
from bigdl.chronos.detector.anomaly.util import roll_arr, scale_arr
import numpy as np

def create_tf_model(compress_rate, input_dim, optimizer='adadelta', loss='binary_crossentropy', lr=0.001):
    if False:
        i = 10
        return i + 15
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend
    inp = Input(shape=(input_dim,))
    encoded = Dense(int(compress_rate * input_dim), activation='relu')(inp)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    backend.set_value(autoencoder.optimizer.learning_rate, lr)
    return autoencoder

def create_torch_model(compress_rate, input_dim):
    if False:
        print('Hello World!')
    import torch.nn as nn
    autoencoder = nn.Sequential(nn.Linear(input_dim, int(compress_rate * input_dim)), nn.ReLU(), nn.Linear(int(compress_rate * input_dim), input_dim), nn.Sigmoid())
    return autoencoder

class AEDetector(AnomalyDetector):
    """
        Example:
            >>> #The dataset to detect is y
            >>> y = numpy.array(...)
            >>> ad = AEDetector(roll_len=24)
            >>> ad.fit(y)
            >>> anomaly_scores = ad.score()
            >>> anomaly_indexes = ad.anomaly_indexes()
    """

    def __init__(self, roll_len=24, ratio=0.1, compress_rate=0.8, batch_size=100, epochs=200, verbose=0, sub_scalef=1, backend='keras', lr=0.001):
        if False:
            return 10
        '\n        Initialize an AEDetector.\n        AEDetector supports two modes to detect anomalies in input time series.\n\n        1. direct-mode: It trains an autoencoder network directly on the input times series and\n        calculate anomaly scores based on reconstruction error. For each sample\n        in the input, the larger the reconstruction error, the higher the\n        anomaly score.\n\n        2. window mode: It first rolls the input series into a batch of subsequences, each\n        with length = `roll_len`. Then it trains an autoencoder network on the batch of\n        subsequences and calculate the reconstruction error. The anomaly score for each\n        sample is a linear combinition of two parts: 1) the reconstruction error of the\n        sample in a subsequence. 2) the reconstruction error of the entire subsequence\n        as a vector. You can use `sub_scalef` to control the weights of the 2nd part. Note\n        that one sample may belong to several subsequences as subsequences overlap because\n        of rolling, and we only keep the largest anomaly score as the final score.\n\n        :param roll_len: the length of window when rolling the input data. If roll_len=0, direct\n            mode is used. If roll_len >0, window mode is used. When setting roll_len, we suggest\n            use a number that is probably a full or half a cycle in your data. e.g. half a day,\n            one day, etc. Note that roll_len must be smaller than the total length of the input\n            time series.\n        :param ratio: (estimated) ratio of anomalies\n        :param compress_rate: the compression rate of the autoencoder, changing this value will have\n            impact on the reconstruction error it calculated.\n        :param batch_size: batch size for autoencoder training\n        :param epochs: num of epochs fro autoencoder training\n        :param verbose: verbose option for autoencoder training\n        :param sub_scalef: scale factor for the subsequence distance when calculating anomaly score\n        :param backend: the backend type, can be "keras" or "torch"\n        :param lr: the learning rate of model\'s optimizer\n        '
        self.ratio = ratio
        self.compress_rate = compress_rate
        self.roll_len = roll_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.sub_scalef = sub_scalef
        self.recon_err = None
        self.recon_err_subseq = None
        self.anomaly_scores_ = None
        self.backend = backend
        self.lr = lr

    def check_rolled(self, arr):
        if False:
            print('Hello World!')
        if arr.size == 0:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, 'rolled array is empty, please check if roll_len is larger than the total series length')

    def check_data(self, arr):
        if False:
            while True:
                i = 10
        if len(arr.shape) > 1:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, 'Only univariate time series is supported')

    def fit(self, y):
        if False:
            while True:
                i = 10
        '\n        Fit the model.\n\n        :param y: the input time series. y must be 1-D numpy array.\n        '
        self.check_data(y)
        self.anomaly_scores_ = np.zeros_like(y)
        if self.roll_len != 0:
            y = roll_arr(y, self.roll_len)
            self.check_rolled(y)
        else:
            y = y.reshape(1, -1)
            self.check_rolled(y)
        y = scale_arr(y)
        if self.backend == 'keras':
            ae_model = create_tf_model(self.compress_rate, len(y[0]), lr=self.lr)
            ae_model.fit(y, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
            y_pred = ae_model.predict(y)
        elif self.backend == 'torch':
            import torch.optim as optim
            import torch.nn as nn
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            ae_model = create_torch_model(self.compress_rate, len(y[0]))
            optimizer = optim.Adadelta(ae_model.parameters(), lr=self.lr)
            criterion = nn.BCELoss()
            y = torch.from_numpy(y).float()
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            train_loader = DataLoader(TensorDataset(y, y), batch_size=int(self.batch_size), shuffle=True)
            for epochs in range(self.epochs):
                for (x_batch, y_batch) in train_loader:
                    optimizer.zero_grad()
                    yhat = ae_model(x_batch)
                    loss = criterion(yhat, y_batch)
                    loss.backward()
                    optimizer.step()
            y_pred_list = []
            for (x_batch, y_batch) in train_loader:
                y_pred_list.append(ae_model(x_batch).detach().numpy())
            y_pred = np.concatenate(y_pred_list, axis=0)
        else:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, "backend type can only be 'keras' or 'torch'")
        self.recon_err = abs(y - y_pred)
        if self.roll_len != 0:
            self.recon_err_subseq = np.linalg.norm(self.recon_err, axis=1)

    def score(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets the anomaly scores for each sample.\n        All anomaly scores are positive numbers. Samples with larger scores are more\n        likely the anomalies.\n        If rolled, the anomaly score is calculated by aggregating the reconstruction\n        errors of each point and subsequence.\n\n        :return: the anomaly scores, in an array format with the same size as input\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.anomaly_scores_ is None:
            invalidInputError(False, 'please call fit before calling score')
        if self.recon_err_subseq is not None:
            for (index, e) in np.ndenumerate(self.recon_err):
                agg_err = e + self.sub_scalef * self.recon_err_subseq[index[0]]
                y_index = index[0] + index[1]
                if agg_err > self.anomaly_scores_[y_index]:
                    self.anomaly_scores_[y_index] = agg_err
        else:
            self.anomaly_scores_ = self.recon_err
        self.anomaly_scores_ = scale_arr(self.anomaly_scores_.reshape(-1, 1)).squeeze()
        return self.anomaly_scores_

    def anomaly_indexes(self):
        if False:
            return 10
        '\n        Gets the indexes of N samples with the largest anomaly scores in y\n        (N = size of input y * AEDetector.ratio)\n\n        :return: the indexes of N samples\n        '
        if self.anomaly_scores_ is None:
            self.score()
        num_anomalies = int(len(self.anomaly_scores_) * self.ratio)
        return self.anomaly_scores_.argsort()[-num_anomalies:]