from bigdl.chronos.forecaster.tf.base_forecaster import BaseTF2Forecaster
from bigdl.chronos.model.tf2.TCN_keras import model_creator, TemporalBlock, TemporalConvNet

class TCNForecaster(BaseTF2Forecaster):
    """
    Example:
        >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
        >>> forecaster = TCNForecaster(past_seq_len=24,
                                       future_seq_len=5,
                                       input_feature_num=1,
                                       output_feature_num=1,
                                       ...)
        >>> forecaster.fit((x_train, y_train))
        >>> forecaster.save({ckpt_name})
        >>> forecaster.load({ckpt_name})
    """

    def __init__(self, past_seq_len, future_seq_len, input_feature_num, output_feature_num, num_channels=[16] * 3, kernel_size=3, normalization=False, repo_initialization=True, dropout=0.1, optimizer='Adam', loss='mse', lr=0.001, metrics=['mse'], seed=None, distributed=False, workers_per_node=1, distributed_backend='ray'):
        if False:
            return 10
        '\n        Build a TCN Forecast Model.\n\n        TCN Forecast may fall into local optima. Please set repo_initialization\n        to False to alleviate the issue. You can also change a random seed to\n        work around.\n\n        :param past_seq_len: Specify the history time steps (i.e. lookback).\n        :param future_seq_len: Specify the output time steps (i.e. horizon).\n        :param input_feature_num: Specify the feature dimension.\n        :param output_feature_num: Specify the output dimension.\n        :param num_channels: Specify the convolutional layer filter number in\n               TCN\'s encoder. This value defaults to [16]*3.\n        :param kernel_size: Specify convolutional layer filter height in TCN\'s\n               encoder. This value defaults to 3.\n        :param normalization: bool, Specify if to use normalization trick to\n               alleviate distribution shift. It first subtractes the last value\n               of the sequence and add back after the model forwarding.\n               This value defaults to False.\n        :param repo_initialization: if to use framework default initialization,\n               True to use paper author\'s initialization and False to use the\n               framework\'s default initialization. The value defaults to True.\n        :param dropout: Specify the dropout close possibility (i.e. the close\n               possibility to a neuron). This value defaults to 0.1.\n        :param optimizer: Specify the optimizer used for training. This value\n               defaults to "Adam".\n        :param loss: Str or a tf.keras.losses.Loss instance, specify the loss function\n               used for training. This value defaults to "mse". You can choose\n               from "mse", "mae" and "huber_loss" or any customized loss instance\n               you want to use.\n        :param lr: Specify the learning rate. This value defaults to 0.001.\n        :param metrics: A list contains metrics for evaluating the quality of\n               forecasting. You may only choose from "mse" and "mae" for a\n               distributed forecaster. You may choose from "mse", "mae",\n               "rmse", "r2", "mape", "smape" or a callable function for a\n               non-distributed forecaster. If callable function, it signature\n               should be func(y_true, y_pred), where y_true and y_pred are numpy\n               ndarray.\n        :param seed: int, random seed for training. This value defaults to None.\n        :param distributed: bool, if init the forecaster in a distributed\n               fashion. If True, the internal model will use an Orca Estimator.\n               If False, the internal model will use a pytorch model. The value\n               defaults to False.\n        :param workers_per_node: int, the number of worker you want to use.\n               The value defaults to 1. The param is only effective when\n               distributed is set to True.\n        :param distributed_backend: str, select from "ray" or\n               "horovod". The value defaults to "ray".\n        '
        self.model_config = {'past_seq_len': past_seq_len, 'future_seq_len': future_seq_len, 'input_feature_num': input_feature_num, 'output_feature_num': output_feature_num, 'num_channels': num_channels, 'kernel_size': kernel_size, 'repo_initialization': repo_initialization, 'dropout': dropout, 'loss': loss, 'lr': lr, 'optim': optimizer, 'normalization': normalization}
        self.model_creator = model_creator
        self.custom_objects_config = {'TemporalBlock': TemporalBlock, 'TemporalConvNet': TemporalConvNet}
        self.distributed = distributed
        self.local_distributed_backend = 'subprocess'
        self.remote_distributed_backend = distributed_backend
        self.workers_per_node = workers_per_node
        self.lr = lr
        self.metrics = metrics
        self.seed = seed
        self.quantize_available = True
        super().__init__()