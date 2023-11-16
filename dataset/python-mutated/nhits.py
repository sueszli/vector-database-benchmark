"""
N-HiTS
------
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule, io_processor
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout
logger = get_logger(__name__)
ACTIVATIONS = ['ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid', 'GELU']

class _Block(nn.Module):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, num_layers: int, layer_width: int, nr_params: int, pooling_kernel_size: int, n_freq_downsample: int, batch_norm: bool, dropout: float, activation: str, MaxPool1d: bool):
        if False:
            i = 10
            return i + 15
        'PyTorch module implementing the basic building block of the N-HiTS architecture.\n\n        The blocks produce outputs of size (target_length, nr_params); i.e.\n        "one vector per parameter". The parameters are predicted only for forecast outputs.\n        Backcast outputs are in the original "domain".\n\n        Parameters\n        ----------\n        input_chunk_length\n            The length of the input sequence fed to the model.\n        output_chunk_length\n            The length of the forecast of the model.\n        num_layers\n            The number of fully connected layers preceding the final forking layers.\n        layer_width\n            The number of neurons that make up each fully connected layer.\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used)\n        pooling_kernel_size\n            The kernel size for the initial pooling layer\n        n_freq_downsample\n            The factor by which to downsample time at the output (before interpolating)\n        batch_norm\n            Whether to use batch norm\n        dropout\n            Dropout probability\n        activation\n            The activation function of encoder/decoder intermediate layer.\n        MaxPool1d\n            Use MaxPool1d pooling. False uses AvgPool1d\n        Inputs\n        ------\n        x of shape `(batch_size, input_chunk_length)`\n            Tensor containing the input sequence.\n\n        Outputs\n        -------\n        x_hat of shape `(batch_size, input_chunk_length)`\n            Tensor containing the \'backcast\' of the block, which represents an approximation of `x`\n            given the constraints of the functional space determined by `g`.\n        y_hat of shape `(batch_size, output_chunk_length)`\n            Tensor containing the forward forecast of the block.\n\n        '
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params
        self.pooling_kernel_size = pooling_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.MaxPool1d = MaxPool1d
        raise_if_not(activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}")
        self.activation = getattr(nn, activation)()
        '\n        Note:\n        -----\n        We use two "last" layers, one for the backcast yielding n_theta_backcast outputs,\n        and one for the forecast yielding n_theta_forecast outputs.\n\n        In the original code, only one last layer yielding "input_chunk_length + n_theta_forecast" [1]\n        outputs is used. So they don\'t use interpolation for the backcast [2], contrary to what is\n        explained in the paper. Here we use what is explained in the paper.\n\n        [1] https://github.com/cchallu/n-hits/blob/4e929ed31e1d3ff5169b4aa0d3762a0040abb8db/\n        src/models/nhits/nhits.py#L263\n        [2] https://github.com/cchallu/n-hits/blob/4e929ed31e1d3ff5169b4aa0d3762a0040abb8db/\n        src/models/nhits/nhits.py#L66\n        '
        n_theta_backcast = max(input_chunk_length // n_freq_downsample, 1)
        n_theta_forecast = max(output_chunk_length // n_freq_downsample, 1)
        pool1d = nn.MaxPool1d if self.MaxPool1d else nn.AvgPool1d
        self.pooling_layer = pool1d(kernel_size=self.pooling_kernel_size, stride=self.pooling_kernel_size, ceil_mode=True)
        in_len = int(np.ceil(input_chunk_length / pooling_kernel_size))
        self.layer_widths = [in_len] + [self.layer_width] * self.num_layers
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(in_features=self.layer_widths[i], out_features=self.layer_widths[i + 1]))
            layers.append(self.activation)
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(num_features=self.layer_widths[i + 1]))
            if self.dropout > 0:
                layers.append(MonteCarloDropout(p=self.dropout))
        self.layers = nn.Sequential(*layers)
        self.backcast_linear_layer = nn.Linear(in_features=layer_width, out_features=n_theta_backcast)
        self.forecast_linear_layer = nn.Linear(in_features=layer_width, out_features=nr_params * n_theta_forecast)

    def forward(self, x):
        if False:
            while True:
                i = 10
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.pooling_layer(x)
        x = x.squeeze(1)
        x = self.layers(x)
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)
        theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)
        theta_backcast = theta_backcast.unsqueeze(1)
        x_hat = F.interpolate(theta_backcast, size=self.input_chunk_length, mode='linear')
        y_hat = F.interpolate(theta_forecast, size=self.output_chunk_length, mode='linear')
        x_hat = x_hat.squeeze(1)
        y_hat = y_hat.reshape(x.shape[0], self.output_chunk_length, self.nr_params)
        return (x_hat, y_hat)

class _Stack(nn.Module):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, num_blocks: int, num_layers: int, layer_width: int, nr_params: int, pooling_kernel_sizes: Tuple[int], n_freq_downsample: Tuple[int], batch_norm: bool, dropout: float, activation: str, MaxPool1d: bool):
        if False:
            print('Hello World!')
        "PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.\n\n        Parameters\n        ----------\n        input_chunk_length\n            The length of the input sequence fed to the model.\n        output_chunk_length\n            The length of the forecast of the model.\n        num_blocks\n            The number of blocks making up this stack.\n        num_layers\n            The number of fully connected layers preceding the final forking layers in each block.\n        layer_width\n            The number of neurons that make up each fully connected layer in each block.\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used)\n        pooling_kernel_sizes\n            sizes of pooling kernels for every block in this stack\n        n_freq_downsample\n            downsampling factors to apply for block in this stack\n        batch_norm\n            whether to apply batch norm on first block of this stack\n        dropout\n            Dropout probability\n        activation\n            The activation function of encoder/decoder intermediate layer.\n        MaxPool1d\n            Use MaxPool1d pooling. False uses AvgPool1d\n\n        Inputs\n        ------\n        stack_input of shape `(batch_size, input_chunk_length)`\n            Tensor containing the input sequence.\n\n        Outputs\n        -------\n        stack_residual of shape `(batch_size, input_chunk_length)`\n            Tensor containing the 'backcast' of the block, which represents an approximation of `x`\n            given the constraints of the functional space determined by `g`.\n        stack_forecast of shape `(batch_size, output_chunk_length)`\n            Tensor containing the forward forecast of the stack.\n\n        "
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params
        self.blocks_list = [_Block(input_chunk_length, output_chunk_length, num_layers, layer_width, nr_params, pooling_kernel_sizes[i], n_freq_downsample[i], batch_norm=batch_norm and i == 0, dropout=dropout, activation=activation, MaxPool1d=MaxPool1d) for i in range(num_blocks)]
        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        if False:
            while True:
                i = 10
        stack_forecast = torch.zeros(x.shape[0], self.output_chunk_length, self.nr_params, device=x.device, dtype=x.dtype)
        for block in self.blocks_list:
            (x_hat, y_hat) = block(x)
            stack_forecast = stack_forecast + y_hat
            x = x - x_hat
        stack_residual = x
        return (stack_residual, stack_forecast)

class _NHiTSModule(PLPastCovariatesModule):

    def __init__(self, input_dim: int, output_dim: int, nr_params: int, num_stacks: int, num_blocks: int, num_layers: int, layer_widths: List[int], pooling_kernel_sizes: Tuple[Tuple[int]], n_freq_downsample: Tuple[Tuple[int]], batch_norm: bool, dropout: float, activation: str, MaxPool1d: bool, **kwargs):
        if False:
            return 10
        'PyTorch module implementing the N-HiTS architecture.\n\n        Parameters\n        ----------\n        input_dim\n            The number of input components (target + optional covariates)\n        output_dim\n            Number of output components in the target\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used).\n        num_stacks\n            The number of stacks that make up the whole model.\n        num_blocks\n            The number of blocks making up every stack.\n        num_layers\n            The number of fully connected layers preceding the final forking layers in each block of every stack.\n        layer_widths\n            Determines the number of neurons that make up each fully connected layer in each block of every stack.\n            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds\n            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks\n            with FC layers of the same width.\n        pooling_kernel_sizes\n            size of pooling kernels for every stack and every block\n        n_freq_downsample\n            downsampling factors to apply for every stack and every block\n        batch_norm\n            Whether to apply batch norm on first block of the first stack\n        dropout\n            Dropout probability\n        activation\n            The activation function of encoder/decoder intermediate layer.\n        MaxPool1d\n            Use MaxPool1d pooling. False uses AvgPool1d\n        **kwargs\n            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.\n\n        Inputs\n        ------\n        x of shape `(batch_size, input_chunk_length)`\n            Tensor containing the input sequence.\n\n        Outputs\n        -------\n        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`\n            Tensor containing the output of the NBEATS module.\n\n        '
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.input_chunk_length_multi = self.input_chunk_length * input_dim
        self.output_chunk_length_multi = self.output_chunk_length * input_dim
        self.stacks_list = [_Stack(self.input_chunk_length_multi, self.output_chunk_length_multi, num_blocks, num_layers, layer_widths[i], nr_params, pooling_kernel_sizes[i], n_freq_downsample[i], batch_norm=batch_norm and i == 0, dropout=dropout, activation=activation, MaxPool1d=MaxPool1d) for i in range(num_stacks)]
        self.stacks = nn.ModuleList(self.stacks_list)
        self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)

    @io_processor
    def forward(self, x_in: Tuple):
        if False:
            print('Hello World!')
        (x, _) = x_in
        x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        x = x.squeeze(dim=2)
        y = torch.zeros(x.shape[0], self.output_chunk_length_multi, self.nr_params, device=x.device, dtype=x.dtype)
        for stack in self.stacks_list:
            (stack_residual, stack_forecast) = stack(x)
            y = y + stack_forecast
            x = stack_residual
        y = y.view(y.shape[0], self.output_chunk_length, self.input_dim, self.nr_params)[:, :, :self.output_dim, :]
        return y

class NHiTSModel(PastCovariatesTorchModel):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, num_stacks: int=3, num_blocks: int=1, num_layers: int=2, layer_widths: Union[int, List[int]]=512, pooling_kernel_sizes: Optional[Tuple[Tuple[int]]]=None, n_freq_downsample: Optional[Tuple[Tuple[int]]]=None, dropout: float=0.1, activation: str='ReLU', MaxPool1d: bool=True, **kwargs):
        if False:
            while True:
                i = 10
        'An implementation of the N-HiTS model, as presented in [1]_.\n\n        N-HiTS is similar to N-BEATS (implemented in :class:`NBEATSModel`),\n        but attempts to provide better performance at lower computational cost by introducing\n        multi-rate sampling of the inputs and multi-scale interpolation of the outputs.\n\n        Similar to :class:`NBEATSModel`, in addition to the univariate version presented in the paper,\n        this implementation also supports multivariate series (and covariates) by flattening the model inputs\n        to a 1-D series and reshaping the outputs to a tensor of appropriate dimensions. Furthermore, it also\n        supports producing probabilistic forecasts (by specifying a `likelihood` parameter).\n\n        This model supports past covariates (known for `input_chunk_length` points before prediction time).\n\n        The multi-rate sampling is done via MaxPooling, which is controlled by ``pooling_kernel_sizes``.\n        This parameter can be a tuple of tuples, of size (num_stacks x num_blocks), specifying the kernel\n        size for each block in each stack. If left to ``None``, some default values will be used based on\n        ``input_chunk_length``.\n        Similarly, the multi-scale interpolation is controlled by ``n_freq_downsample``, which gives the\n        downsampling factors to be used in each block of each stack. If left to ``None``, some default\n        values will be used based on the ``output_chunk_length``.\n\n        Parameters\n        ----------\n        input_chunk_length\n            The length of the input sequence fed to the model.\n        output_chunk_length\n            The length of the forecast of the model.\n        num_stacks\n            The number of stacks that make up the whole model.\n        num_blocks\n            The number of blocks making up every stack.\n        num_layers\n            The number of fully connected layers preceding the final forking layers in each block of every stack.\n        layer_widths\n            Determines the number of neurons that make up each fully connected layer in each block of every stack.\n            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds\n            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks\n            with FC layers of the same width.\n        pooling_kernel_sizes\n            If set, this parameter must be a tuple of tuples, of size (num_stacks x num_blocks), specifying the kernel\n            size for each block in each stack used for the input pooling layer.\n            If left to ``None``, some default values will be used based on ``input_chunk_length``.\n        n_freq_downsample\n            If set, this parameter must be a tuple of tuples, of size (num_stacks x num_blocks), specifying the\n            downsampling factors before interpolation, for each block in each stack.\n            If left to ``None``, some default values will be used based on ``output_chunk_length``.\n        dropout\n            The dropout probability to be used in fully connected layers. This is compatible with Monte Carlo dropout\n            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at\n            prediction time).\n        activation\n            The activation function of encoder/decoder intermediate layer (default=\'ReLU\').\n            Supported activations: [\'ReLU\',\'RReLU\', \'PReLU\', \'Softplus\', \'Tanh\', \'SELU\', \'LeakyReLU\',  \'Sigmoid\']\n        MaxPool1d\n            Use MaxPool1d pooling. False uses AvgPool1d\n        **kwargs\n            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and\n            Darts\' :class:`TorchForecastingModel`.\n\n        loss_fn\n            PyTorch loss function used for training.\n            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.\n            Default: ``torch.nn.MSELoss()``.\n        likelihood\n            One of Darts\' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for\n            probabilistic forecasts. Default: ``None``.\n        torch_metrics\n            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found\n            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.\n        optimizer_cls\n            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.\n        optimizer_kwargs\n            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{\'lr\': 1e-3}``\n            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``\n            will be used. Default: ``None``.\n        lr_scheduler_cls\n            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds\n            to using a constant learning rate. Default: ``None``.\n        lr_scheduler_kwargs\n            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.\n        use_reversible_instance_norm\n            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.\n            It is only applied to the features of the target series and not the covariates.\n        batch_size\n            Number of time series (input and output sequences) used in each training pass. Default: ``32``.\n        n_epochs\n            Number of epochs over which to train the model. Default: ``100``.\n        model_name\n            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,\n            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part\n            of the name is formatted with the local date and time, while PID is the processed ID (preventing models\n            spawned at the same time by different processes to share the same model_name). E.g.,\n            ``"2021-06-14_09_53_32_torch_model_run_44607"``.\n        work_dir\n            Path of the working directory, where to save checkpoints and Tensorboard summaries.\n            Default: current working directory.\n        log_tensorboard\n            If set, use Tensorboard to log the different parameters. The logs will be located in:\n            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.\n        nr_epochs_val_period\n            Number of epochs to wait before evaluating the validation loss (if a validation\n            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.\n        force_reset\n            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will\n            be discarded). Default: ``False``.\n        save_checkpoints\n            Whether or not to automatically save the untrained model and checkpoints from training.\n            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where\n            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,\n            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using\n            :func:`save()` and loaded using :func:`load()`. Default: ``False``.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        random_state\n            Control the randomness of the weights initialization. Check this\n            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.\n            Default: ``None``.\n        pl_trainer_kwargs\n            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets\n            that performs the training, validation and prediction processes. These presets include automatic\n            checkpointing, tensorboard logging, setting the torch device and more.\n            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer\n            object. Check the `PL Trainer documentation\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the\n            supported kwargs. Default: ``None``.\n            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",\n            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``\n            dict:\n\n\n            - ``{"accelerator": "cpu"}`` for CPU,\n            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),\n            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.\n\n            For more info, see here:\n            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and\n            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus\n\n            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts\'\n            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.\n            The model will stop training early if the validation loss `val_loss` does not improve beyond\n            specifications. For more information on callbacks, visit:\n            `PyTorch Lightning Callbacks\n            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from pytorch_lightning.callbacks.early_stopping import EarlyStopping\n\n                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over\n                # a period of 5 epochs (`patience`)\n                my_stopper = EarlyStopping(\n                    monitor="val_loss",\n                    patience=5,\n                    min_delta=0.05,\n                    mode=\'min\',\n                )\n\n                pl_trainer_kwargs={"callbacks": [my_stopper]}\n            ..\n\n            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional\n            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.\n        show_warnings\n            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of\n            your forecasting use case. Default: ``False``.\n\n        References\n        ----------\n        .. [1] C. Challu et al. "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting",\n               https://arxiv.org/abs/2201.12886\n        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import NHiTSModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series[\'p (mbar)\'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series[\'rain (mm)\'][:100]\n        >>> # increasing the number of blocks\n        >>> model = NHiTSModel(\n        >>>     input_chunk_length=6,\n        >>>     output_chunk_length=6,\n        >>>     num_blocks=2,\n        >>>     n_epochs=5,\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[958.2354389 ],\n               [939.23201079],\n               [987.51425784],\n               [919.41209025],\n               [925.09583093],\n               [938.95625528]])\n        '
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        raise_if_not(isinstance(layer_widths, int) or len(layer_widths) == num_stacks, 'Please pass an integer or a list of integers with length `num_stacks`as value for the `layer_widths` argument.', logger)
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.activation = activation
        self.MaxPool1d = MaxPool1d
        self.batch_norm = False
        self.dropout = dropout
        sizes = self._prepare_pooling_downsampling(pooling_kernel_sizes, n_freq_downsample, self.input_chunk_length, self.output_chunk_length, num_blocks, num_stacks)
        (self.pooling_kernel_sizes, self.n_freq_downsample) = sizes
        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * self.num_stacks

    @property
    def supports_multivariate(self) -> bool:
        if False:
            return 10
        return True

    @staticmethod
    def _prepare_pooling_downsampling(pooling_kernel_sizes, n_freq_downsample, in_len, out_len, num_blocks, num_stacks):
        if False:
            while True:
                i = 10

        def _check_sizes(tup, name):
            if False:
                return 10
            raise_if_not(len(tup) == num_stacks, f'the length of {name} must match the number of stacks.')
            raise_if_not(all([len(i) == num_blocks for i in tup]), f'the length of each tuple in {name} must be `num_blocks={num_blocks}`')
        if pooling_kernel_sizes is None:
            max_v = max(in_len // 2, 1)
            pooling_kernel_sizes = tuple(((int(v),) * num_blocks for v in max_v // np.geomspace(1, max_v, num_stacks)))
            logger.info(f'(N-HiTS): Using automatic kernel pooling size: {pooling_kernel_sizes}.')
        else:
            _check_sizes(pooling_kernel_sizes, '`pooling_kernel_sizes`')
        if n_freq_downsample is None:
            max_v = max(out_len // 2, 1)
            n_freq_downsample = tuple(((int(v),) * num_blocks for v in max_v // np.geomspace(1, max_v, num_stacks)))
            logger.info(f'(N-HiTS):  Using automatic downsampling coefficients: {n_freq_downsample}.')
        else:
            _check_sizes(n_freq_downsample, '`n_freq_downsample`')
            raise_if_not(n_freq_downsample[-1][-1] == 1, 'the downsampling coefficient of the last block of the last stack must be 1 ' + '(i.e., `n_freq_downsample[-1][-1]`).')
        return (pooling_kernel_sizes, n_freq_downsample)

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        if False:
            for i in range(10):
                print('nop')
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return _NHiTSModule(input_dim=input_dim, output_dim=output_dim, nr_params=nr_params, num_stacks=self.num_stacks, num_blocks=self.num_blocks, num_layers=self.num_layers, layer_widths=self.layer_widths, pooling_kernel_sizes=self.pooling_kernel_sizes, n_freq_downsample=self.n_freq_downsample, batch_norm=self.batch_norm, dropout=self.dropout, activation=self.activation, MaxPool1d=self.MaxPool1d, **self.pl_module_params)