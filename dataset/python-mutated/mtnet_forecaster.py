from bigdl.chronos.model.tf2.MTNet_keras import MTNetKeras as MTNetKerasModel
from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.metric.forecast_metrics import Evaluator

class MTNetForecaster(Forecaster):
    """
        Example:
            >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
            >>> model = MTNetForecaster(target_dim=1,
                                        feature_dim=x_train.shape[-1],
                                        long_series_num=6,
                                        series_length=2
                                        )
            >>> model.fit(data=(x_train, y_train),
                          validation_data=(x_val, y_val),
                          epochs=2,
                          batch_size=32)
            >>> model.predict(x_test, batch_size=batch_size)
    """

    def __init__(self, target_dim=1, feature_dim=1, long_series_num=1, series_length=1, ar_window_size=1, cnn_height=1, cnn_hid_size=32, rnn_hid_sizes=[16, 32], lr=0.001, loss='mae', cnn_dropout=0.2, rnn_dropout=0.2, metric='mean_squared_error', uncertainty: bool=False):
        if False:
            print('Hello World!')
        '\n        Build a MTNet Forecast Model.\n\n        :param target_dim: the dimension of model output\n        :param feature_dim: the dimension of input feature\n        :param long_series_num: the number of series for the long-term memory series\n        :param series_length: the series size for long-term and short-term memory series\n        :param ar_window_size: the auto regression window size in MTNet\n        :param cnn_hid_size: the hidden layer unit for cnn in encoder\n        :param rnn_hid_sizes: the hidden layers unit for rnn in encoder\n        :param cnn_height: cnn filter height in MTNet\n        :param metric: the metric for validation and evaluation\n        :param uncertainty: whether to enable calculation of uncertainty\n        :param lr: learning rate\n        :param loss: the target function you want to optimize on\n        :param cnn_dropout: the dropout possibility for cnn in encoder\n        :param rnn_dropout: the dropout possibility for rnn in encoder\n        '
        self.check_optional_config = False
        self.mc = uncertainty
        self.model_config = {'feature_num': feature_dim, 'output_dim': target_dim, 'metrics': [metric], 'mc': uncertainty, 'time_step': series_length, 'long_num': long_series_num, 'ar_window': ar_window_size, 'cnn_height': cnn_height, 'past_seq_len': (long_series_num + 1) * series_length, 'cnn_hid_size': cnn_hid_size, 'rnn_hid_sizes': rnn_hid_sizes, 'lr': lr, 'cnn_dropout': cnn_dropout, 'rnn_dropout': rnn_dropout, 'loss': loss}
        super().__init__()
        self._fitted = False
        self.internal = MTNetKerasModel(check_optional_config=self.check_optional_config)

    def fit(self, data, epochs=2, batch_size=32, validation_data=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        :param data: The data support following formats:\n\n            | 1. a numpy ndarray tuple (x, y):\n            | x's shape is (num_samples, (long_series_num+1)*series_length, feature_dim)\n            | y's shape is (num_samples, target_dim)\n\n        :param epochs: Number of epochs you want to train. The value defaults to 2.\n        :param batch_size: Number of batch size you want to train. The value defaults to 32.\n        :param validation_data: Data on which to evaluate the loss\n               and any model metrics at the end of each epoch.\n               The model will not be trained on this data.\n        "
        self.model_config.update({'batch_size': batch_size})
        self.internal.fit_eval(data, validation_data=validation_data, epochs=epochs, verbose=1, **self.model_config)
        self._fitted = True

    def predict(self, data, batch_size=32):
        if False:
            print('Hello World!')
        "\n        :param data: The data support following formats:\n\n               | 1. data's shape is (num_samples, (long_series_num+1)*series_length, feature_dim)\n\n        :param batch_size: predict batch size. The value will not affect evaluate\n               result but will affect resources cost(e.g. memory and time).\n\n        :return: A numpy.ndarray with shape of (num_samples, feature_dum).\n        "
        from bigdl.nano.utils.common import invalidInputError
        if not self._fitted:
            invalidInputError(False, 'You must call fit or restore first before calling predict!')
        yhat = self.internal.predict(data, batch_size=batch_size)
        return yhat

    def evaluate(self, data, metric=['mae'], batch_size=32, multioutput='raw_values'):
        if False:
            print('Hello World!')
        '\n        :param data: The data support following formats:\n\n            | 1. a numpy ndarray tuple (x, y):\n            | x\'s shape is (num_samples, (long_series_num+1)*series_length, feature_dim)\n            | y\'s shape is (num_samples, target_dim)\n\n        :param metric:  metric is the evaluation metric name to optimize, e.g. ["mae"].\n        :param batch_size: evaluate batch size. The value will not affect evaluate\n               result but will affect resources cost(e.g. memory and time).\n        :param multioutput: Defines aggregating of multiple output values.\n               String in [\'raw_values\', \'uniform_average\']. The value defaults to\n               \'raw_values\'.\n\n        :return: A list of evaluation results. Calculation results for each metrics.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if not self._fitted:
            invalidInputError(False, 'You must call fit or restore first before calling evaluate!')
        yhat = self.internal.predict(data[0], batch_size=batch_size)
        return [Evaluator.evaluate(m, y_true=data[1], y_pred=yhat, aggregate=multioutput) for m in metric]

    def save(self, checkpoint_file):
        if False:
            i = 10
            return i + 15
        '\n        Save the forecaster.\n\n        :param checkpoint_file: The location you want to save the forecaster.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if not self.fitted:
            invalidInputError(False, 'You must call fit or restore first before calling save!')
        self.internal.save(checkpoint_file=checkpoint_file)

    def load(self, checkpoint_file):
        if False:
            print('Hello World!')
        '\n        Load the forecaster.\n\n        :param checkpoint_file: The checkpoint file location you want to load the forecaster.\n        '
        self.internal.restore(checkpoint_file=checkpoint_file, **self.model_config)