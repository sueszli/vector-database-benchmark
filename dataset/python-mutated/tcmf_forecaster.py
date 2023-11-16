from bigdl.chronos.model.tcmf_model import TCMFNdarrayModelWrapper, TCMFXshardsModelWrapper
from bigdl.orca.data import SparkXShards
from bigdl.chronos.forecaster.abstract import Forecaster

class TCMFForecaster(Forecaster):
    """
        Example:
            >>> import numpy as np
            >>> model = TCMFForecaster()
            >>> fit_params = dict(val_len=12,
                               start_date="2020-1-1",
                               freq="5min",
                               y_iters=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)
            >>> ndarray_input = {'id': np.arange(300), 'y': np.random.rand(300, 480)}
            >>> model.fit(ndarray_input, fit_params)
            >>> horizon = np.random.randint(1, 50)
            >>> yhat = model.predict(horizon=horizon)
            >>> model.save({tempdirname})
            >>> loaded_model = TCMFForecaster.load({tempdirname}, is_xshards_distributed=False)
            >>> data_new = np.random.rand(300, horizon)
            >>> model.evaluate(target_value=dict({"y": data_new}), metric=['mse'])
            >>> model.fit_incremental({"y": data_new})
            >>> yhat_incr = model.predict(horizon=horizon)
    """

    def __init__(self, vbsize=128, hbsize=256, num_channels_X=[32, 32, 32, 32, 32, 1], num_channels_Y=[16, 16, 16, 16, 16, 1], kernel_size=7, dropout=0.1, rank=64, kernel_size_Y=7, learning_rate=0.0005, normalize=False, use_time=True, svd=True):
        if False:
            i = 10
            return i + 15
        '\n        Build a TCMF Forecast Model.\n\n        :param vbsize: int, default is 128.\n            Vertical batch size, which is the number of cells per batch.\n        :param hbsize: int, default is 256.\n            Horizontal batch size, which is the number of time series per batch.\n        :param num_channels_X: list, default=[32, 32, 32, 32, 32, 1].\n            List containing channel progression of temporal convolution network for local model\n        :param num_channels_Y: list, default=[16, 16, 16, 16, 16, 1]\n            List containing channel progression of temporal convolution network for hybrid model.\n        :param kernel_size: int, default is 7.\n            Kernel size for local models\n        :param dropout: float, default is 0.1.\n            Dropout rate during training\n        :param rank: int, default is 64.\n            The rank in matrix factorization of global model.\n        :param kernel_size_Y: int, default is 7.\n            Kernel size of hybrid model\n        :param learning_rate: float, default is 0.0005\n        :param normalize: boolean, false by default.\n            Whether to normalize input data for training.\n        :param use_time: boolean, default is True.\n            Whether to use time coveriates.\n        :param svd: boolean, default is False.\n            Whether factor matrices are initialized by NMF\n        '
        self.internal = None
        self.config = {'vbsize': vbsize, 'hbsize': hbsize, 'num_channels_X': num_channels_X, 'num_channels_Y': num_channels_Y, 'kernel_size': kernel_size, 'dropout': dropout, 'rank': rank, 'kernel_size_Y': kernel_size_Y, 'learning_rate': learning_rate, 'normalize': normalize, 'use_time': use_time, 'svd': svd}

    def fit(self, x, val_len=24, start_date='2020-4-1', freq='1H', covariates=None, dti=None, period=24, y_iters=10, init_FX_epoch=100, max_FX_epoch=300, max_TCN_epoch=300, alt_iters=10, num_workers=None):
        if False:
            return 10
        '\n        Fit the model on x from scratch\n\n        :param x: the input for fit. Only dict of ndarray and SparkXShards of dict of ndarray\n            are supported. Example: {\'id\': id_arr, \'y\': data_ndarray}, and data_ndarray\n            is of shape (n, T), where n is the number f target time series and T is the\n            number of time steps.\n        :param val_len: int, default is 24.\n            Validation length. We will use the last val_len time points as validation data.\n        :param start_date: str or datetime-like.\n            Start date time for the time-series. e.g. "2020-01-01"\n        :param freq: str or DateOffset, default is \'H\'\n            Frequency of data\n        :param covariates: 2-D ndarray or None. The shape of ndarray should be (r, T), where r is\n            the number of covariates and T is the number of time points.\n            Global covariates for all time series. If None, only default time coveriates will be\n            used while use_time is True. If not, the time coveriates used is the stack of input\n            covariates and default time coveriates.\n        :param dti: DatetimeIndex or None.\n            If None, use default fixed frequency DatetimeIndex generated with start_date and freq.\n        :param period: int, default is 24.\n            Periodicity of input time series, leave it out if not known\n        :param y_iters: int, default is 10.\n            Number of iterations while training the hybrid model.\n        :param init_FX_epoch: int, default is 100.\n            Number of iterations while initializing factors\n        :param max_FX_epoch: int, default is 300.\n            Max number of iterations while training factors.\n        :param max_TCN_epoch: int, default is 300.\n            Max number of iterations while training the local model.\n        :param alt_iters: int, default is 10.\n            Number of iterations while alternate training.\n        :param num_workers: the number of workers you want to use for fit. If None, it defaults to\n            num_ray_nodes in the created OrcaRayContext or 1 if there is no active OrcaRayContext.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.internal is None:
            if isinstance(x, SparkXShards):
                self.internal = TCMFXshardsModelWrapper(self.config)
            elif isinstance(x, dict):
                self.internal = TCMFNdarrayModelWrapper(self.config)
            else:
                invalidInputError(False, 'value of x should be a dict of ndarray or an xShards of dict of ndarray')
            try:
                self.internal.fit(x, num_workers=num_workers, val_len=val_len, start_date=start_date, freq=freq, covariates=covariates, dti=dti, period=period, y_iters=y_iters, init_FX_epoch=init_FX_epoch, max_FX_epoch=max_FX_epoch, max_TCN_epoch=max_TCN_epoch, alt_iters=alt_iters)
            except Exception as inst:
                self.internal = None
                from bigdl.nano.utils.common import invalidOperationError
                invalidOperationError(False, str(inst), cause=inst)
        else:
            invalidInputError(False, 'This model has already been fully trained, you can only run full training once.')

    def fit_incremental(self, x_incr, covariates_incr=None, dti_incr=None):
        if False:
            while True:
                i = 10
        "\n        Incrementally fit the model. Note that we only incrementally fit X_seq (TCN in global model)\n\n        :param x_incr: incremental data to be fitted. It should be of the same format as input x in\n            fit, which is a dict of ndarray or SparkXShards of dict of ndarray.\n            Example: {'id': id_arr, 'y': incr_ndarray}, and incr_ndarray is of shape (n, T_incr)\n            , where n is the number of target time series, T_incr is the number of time steps\n            incremented. You can choose not to input 'id' in x_incr, but if you do, the elements\n            of id in x_incr should be the same as id in x of fit.\n        :param covariates_incr: covariates corresponding to x_incr. 2-D ndarray or None.\n            The shape of ndarray should be (r, T_incr), where r is the number of covariates.\n            Global covariates for all time series. If None, only default time coveriates will be\n            used while use_time is True. If not, the time coveriates used is the stack of input\n            covariates and default time coveriates.\n        :param dti_incr: dti corresponding to the x_incr. DatetimeIndex or None.\n            If None, use default fixed frequency DatetimeIndex generated with the last date of x in\n            fit and freq.\n        "
        self.internal.fit_incremental(x_incr, covariates_incr=covariates_incr, dti_incr=dti_incr)

    def evaluate(self, target_value, metric=['mae'], target_covariates=None, target_dti=None, num_workers=None):
        if False:
            print('Hello World!')
        '\n        Evaluate the model\n\n        :param target_value: target value for evaluation. We interpret its second dimension of\n               as the horizon length for evaluation.\n        :param metric: the metrics. A list of metric names.\n        :param target_covariates: covariates corresponding to target_value.\n            2-D ndarray or None.\n            The shape of ndarray should be (r, horizon), where r is the number of covariates.\n            Global covariates for all time series. If None, only default time coveriates will be\n            used while use_time is True. If not, the time coveriates used is the stack of input\n            covariates and default time coveriates.\n        :param target_dti: dti corresponding to target_value.\n            DatetimeIndex or None.\n            If None, use default fixed frequency DatetimeIndex generated with the last date of x in\n            fit and freq.\n        :param num_workers: the number of workers to use in evaluate. If None, it defaults to\n            num_ray_nodes in the created OrcaRayContext or 1 if there is no active OrcaRayContext.\n\n        :return: A list of evaluation results. Each item represents a metric.\n        '
        return self.internal.evaluate(y=target_value, metric=metric, target_covariates=target_covariates, target_dti=target_dti, num_workers=num_workers)

    def predict(self, horizon=24, future_covariates=None, future_dti=None, num_workers=None):
        if False:
            while True:
                i = 10
        '\n        Predict using a trained forecaster.\n\n        :param horizon: horizon length to look forward.\n        :param future_covariates: covariates corresponding to future horizon steps data to predict.\n            2-D ndarray or None.\n            The shape of ndarray should be (r, horizon), where r is the number of covariates.\n            Global covariates for all time series. If None, only default time coveriates will be\n            used while use_time is True. If not, the time coveriates used is the stack of input\n            covariates and default time coveriates.\n        :param future_dti: dti corresponding to future horizon steps data to predict.\n            DatetimeIndex or None.\n            If None, use default fixed frequency DatetimeIndex generated with the last date of x in\n            fit and freq.\n        :param num_workers: the number of workers to use in predict. If None, it defaults to\n            num_ray_nodes in the created OrcaRayContext or 1 if there is no active OrcaRayContext.\n\n        :return: A numpy ndarray with shape of (nd, horizon), where nd is the same number\n            of time series as input x in fit_eval.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.internal is None:
            invalidInputError(False, 'You should run fit before calling predict()')
        else:
            return self.internal.predict(horizon, future_covariates=future_covariates, future_dti=future_dti, num_workers=num_workers)

    def save(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Save the forecaster.\n\n        :param path: Path to target saved file.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.internal is None:
            invalidInputError(False, 'You should run fit before calling save()')
        else:
            self.internal.save(path)

    def is_xshards_distributed(self):
        if False:
            return 10
        '\n        Check whether model is distributed by input xshards.\n\n        :return: True if the model is distributed by input xshards\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.internal is None:
            invalidInputError(False, 'You should run fit before calling is_xshards_distributed()')
        else:
            return self.internal.is_xshards_distributed()

    @classmethod
    def load(cls, path, is_xshards_distributed=False, minPartitions=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load a saved model.\n\n        :param path: The location you want to save the forecaster.\n        :param is_xshards_distributed: Whether the model is distributed trained with\n            input of dict of SparkXshards.\n        :param minPartitions: The minimum partitions for the XShards.\n\n        :return: the model loaded\n        '
        loaded_model = TCMFForecaster()
        if is_xshards_distributed:
            loaded_model.internal = TCMFXshardsModelWrapper(loaded_model.config)
            loaded_model.internal.load(path, minPartitions=minPartitions)
        else:
            loaded_model.internal = TCMFNdarrayModelWrapper(loaded_model.config)
            loaded_model.internal.load(path)
        return loaded_model