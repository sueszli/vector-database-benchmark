import torch
from bigdl.chronos.forecaster.base_forecaster import BasePytorchForecaster
from bigdl.chronos.model.nbeats_pytorch import model_creator, loss_creator, optimizer_creator

class NBeatsForecaster(BasePytorchForecaster):
    """
    Example:
        >>> # 1. Initialize Forecaster directly
        >>> forecaster = NBeatForecaster(paste_seq_len=10,
                                         future_seq_len=1,
                                         stack_types=("generic", "generic"),
                                         ...)
        >>>
        >>> # 2. The from_tsdataset method can also initialize a NBeatForecaster.
        >>> forecaster.from_tsdataset(tsdata, **kwargs)
        >>> forecaster.fit(tsdata)
        >>> forecaster.to_local() # if you set distributed=True
    """

    def __init__(self, past_seq_len, future_seq_len, stack_types=('generic', 'generic'), nb_blocks_per_stack=3, thetas_dim=(4, 8), share_weights_in_stack=False, hidden_layer_units=256, nb_harmonics=None, optimizer='Adam', loss='mse', lr=0.001, metrics=['mse'], seed=None, distributed=False, workers_per_node=1, distributed_backend='ray'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a NBeats Forecaster Model.\n\n        :param past_seq_len: Specify the history time steps (i.e. lookback).\n        :param future_seq_len: Specify the output time steps (i.e. horizon).\n        :param stack_types: Specifies the type of stack,\n               including "generic", "trend", "seasnoality".\n               This value defaults to ("generic", "generic").\n               If set distributed=True, the second type should not be "generic",\n               use "seasonality" or "trend", e.g. ("generic", "trend").\n        :param nb_blocks_per_stack: Specify the number of blocks\n               contained in each stack, This value defaults to 3.\n        :param thetas_dim: Expansion Coefficients of Multilayer FC Networks.\n               if type is "generic", Extended length factor, if type is "trend"\n               then polynomial coefficients, if type is "seasonality"\n               expressed as a change within each step.\n        :param share_weights_in_stack: Share block weights for each stack.,\n               This value defaults to False.\n        :param hidden_layer_units: Number of fully connected layers with per block.\n               This values defaults to 256.\n        :param nb_harmonics: Only available in "seasonality" type,\n               specifies the time step of backward, This value defaults is None.\n        :param dropout: Specify the dropout close possibility\n               (i.e. the close possibility to a neuron). This value defaults to 0.1.\n        :param optimizer: Specify the optimizer used for training. This value\n               defaults to "Adam".\n        :param loss: str or pytorch loss instance, Specify the loss function\n               used for training. This value defaults to "mse". You can choose\n               from "mse", "mae", "huber_loss" or any customized loss instance\n               you want to use.\n        :param lr: Specify the learning rate. This value defaults to 0.001.\n        :param metrics: A list contains metrics for evaluating the quality of\n               forecasting. You may only choose from "mse" and "mae" for a\n               distributed forecaster. You may choose from "mse", "mae",\n               "rmse", "r2", "mape", "smape" or a callable function for a\n               non-distributed forecaster. If callable function, it signature\n               should be func(y_true, y_pred), where y_true and y_pred are numpy\n               ndarray.\n        :param seed: int, random seed for training. This value defaults to None.\n        :param distributed: bool, if init the forecaster in a distributed\n               fashion. If True, the internal model will use an Orca Estimator.\n               If False, the internal model will use a pytorch model. The value\n               defaults to False.\n        :param workers_per_node: int, the number of worker you want to use.\n               The value defaults to 1. The param is only effective when\n               distributed is set to True.\n        :param distributed_backend: str, select from "ray" or\n               "horovod". The value defaults to "ray".\n        '
        if stack_types[-1] == 'generic' and distributed:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False, "Please set distributed=False or change the type of 'stack_types' to 'trend', 'seasonality', e.g. ('generic', 'seasonality').")
        self.data_config = {'past_seq_len': past_seq_len, 'future_seq_len': future_seq_len, 'input_feature_num': 1, 'output_feature_num': 1}
        self.model_config = {'stack_types': stack_types, 'nb_blocks_per_stack': nb_blocks_per_stack, 'thetas_dim': thetas_dim, 'share_weights_in_stack': share_weights_in_stack, 'hidden_layer_units': hidden_layer_units, 'nb_harmonics': nb_harmonics, 'seed': seed}
        self.loss_config = {'loss': loss}
        self.optim_config = {'lr': lr, 'optim': optimizer}
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        if isinstance(loss, str):
            self.loss_creator = loss_creator
        else:

            def customized_loss_creator(config):
                if False:
                    for i in range(10):
                        print('nop')
                return config['loss']
            self.loss_creator = customized_loss_creator
        self.distributed = distributed
        self.remote_distributed_backend = distributed_backend
        self.local_distributed_backend = 'subprocess'
        self.workers_per_node = workers_per_node
        self.lr = lr
        self.seed = seed
        self.metrics = metrics
        current_num_threads = torch.get_num_threads()
        self.thread_num = current_num_threads
        self.optimized_model_thread_num = current_num_threads
        if current_num_threads >= 24:
            self.num_processes = max(1, current_num_threads // 8)
        else:
            self.num_processes = 1
        self.use_ipex = False
        self.onnx_available = True
        self.quantize_available = True
        self.checkpoint_callback = True
        self.use_hpo = True
        self.optimized_model_output_tensor = True
        super().__init__()

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, future_seq_len=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Build a NBeats Forecaster Model.\n\n        :param tsdataset: Train tsdataset, a bigdl.chronos.data.tsdataset.TSDataset instance.\n        :param past_seq_len: Specify the history time steps (i.e. lookback).\n               Do not specify the 'past_seq_len' if your tsdataset has called\n               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.\n        :param future_seq_len: Specify the output time steps (i.e. horizon).\n               Do not specify the 'future_seq_len' if your tsdataset has called\n               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.\n        :param kwargs: Specify parameters of Forecaster,\n               e.g. loss and optimizer, etc. More info,\n               please refer to NBeatsForecaster.__init__ methods.\n\n        :return: A NBeats Forecaster Model.\n        "
        from bigdl.chronos.data.tsdataset import TSDataset
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(isinstance(tsdataset, TSDataset), f'We only supports input a TSDataset, but get{type(tsdataset)}.')

        def check_time_steps(tsdataset, past_seq_len, future_seq_len):
            if False:
                return 10
            if tsdataset.lookback is not None and past_seq_len is not None:
                future_seq_len = future_seq_len if isinstance(future_seq_len, int) else max(future_seq_len)
                return tsdataset.lookback == past_seq_len and tsdataset.horizon == future_seq_len
            return True
        invalidInputError(not tsdataset._has_generate_agg_feature, "We will add support for 'gen_rolling_feature' method later.")
        if tsdataset.lookback is not None:
            past_seq_len = tsdataset.lookback
            future_seq_len = tsdataset.horizon if isinstance(tsdataset.horizon, int) else max(tsdataset.horizon)
        elif past_seq_len is not None and future_seq_len is not None:
            past_seq_len = past_seq_len if isinstance(past_seq_len, int) else tsdataset.get_cycle_length()
            future_seq_len = future_seq_len if isinstance(future_seq_len, int) else max(future_seq_len)
        else:
            invalidInputError(False, "Forecaster requires 'past_seq_len' and 'future_seq_len' to specify the history time step and output time step.")
        invalidInputError(check_time_steps(tsdataset, past_seq_len, future_seq_len), f'tsdataset already has historical time steps and differs from the given past_seq_len and future_seq_len Expected past_seq_len and future_seq_len to be {(tsdataset.lookback, tsdataset.horizon)}, but found {(past_seq_len, future_seq_len)}', fixMsg='Do not specify past_seq_len and future seq_len or call tsdataset.roll method again and specify time step')
        invalidInputError(not all([tsdataset.id_sensitive, len(tsdataset._id_list) > 1]), 'NBeats only supports univariate forecasting.')
        return cls(past_seq_len=past_seq_len, future_seq_len=future_seq_len, **kwargs)