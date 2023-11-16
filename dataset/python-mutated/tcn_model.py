"""
Temporal Convolutional Network
------------------------------
"""
import math
from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule, io_processor
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.timeseries import TimeSeries
from darts.utils.data import PastCovariatesShiftedDataset
from darts.utils.torch import MonteCarloDropout
logger = get_logger(__name__)

class _ResidualBlock(nn.Module):

    def __init__(self, num_filters: int, kernel_size: int, dilation_base: int, dropout_fn, weight_norm: bool, nr_blocks_below: int, num_layers: int, input_size: int, target_size: int):
        if False:
            i = 10
            return i + 15
        'PyTorch module implementing a residual block module used in `_TCNModule`.\n\n        Parameters\n        ----------\n        num_filters\n            The number of filters in a convolutional layer of the TCN.\n        kernel_size\n            The size of every kernel in a convolutional layer.\n        dilation_base\n            The base of the exponent that will determine the dilation on every level.\n        dropout_fn\n            The dropout function to be applied to every convolutional layer.\n        weight_norm\n            Boolean value indicating whether to use weight normalization.\n        nr_blocks_below\n            The number of residual blocks before the current one.\n        num_layers\n            The number of convolutional layers.\n        input_size\n            The dimensionality of the input time series of the whole network.\n        target_size\n            The dimensionality of the output time series of the whole network.\n\n        Inputs\n        ------\n        x of shape `(batch_size, in_dimension, input_chunk_length)`\n            Tensor containing the features of the input sequence.\n            in_dimension is equal to `input_size` if this is the first residual block,\n            in all other cases it is equal to `num_filters`.\n\n        Outputs\n        -------\n        y of shape `(batch_size, out_dimension, input_chunk_length)`\n            Tensor containing the output sequence of the residual block.\n            out_dimension is equal to `output_size` if this is the last residual block,\n            in all other cases it is equal to `num_filters`.\n        '
        super().__init__()
        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below
        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, dilation=dilation_base ** nr_blocks_below)
        self.conv2 = nn.Conv1d(num_filters, output_dim, kernel_size, dilation=dilation_base ** nr_blocks_below)
        if weight_norm:
            (self.conv1, self.conv2) = (nn.utils.weight_norm(self.conv1), nn.utils.weight_norm(self.conv2))
        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        if False:
            while True:
                i = 10
        residual = x
        left_padding = self.dilation_base ** self.nr_blocks_below * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual
        return x

class _TCNModule(PLPastCovariatesModule):

    def __init__(self, input_size: int, kernel_size: int, num_filters: int, num_layers: Optional[int], dilation_base: int, weight_norm: bool, target_size: int, nr_params: int, target_length: int, dropout: float, **kwargs):
        if False:
            print('Hello World!')
        "PyTorch module implementing a dilated TCN module used in `TCNModel`.\n\n\n        Parameters\n        ----------\n        input_size\n            The dimensionality of the input time series.\n        target_size\n            The dimensionality of the output time series.\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used).\n        target_length\n            Number of time steps the torch module will predict into the future at once.\n        kernel_size\n            The size of every kernel in a convolutional layer.\n        num_filters\n            The number of filters in a convolutional layer of the TCN.\n        num_layers\n            The number of convolutional layers.\n        weight_norm\n            Boolean value indicating whether to use weight normalization.\n        dilation_base\n            The base of the exponent that will determine the dilation on every level.\n        dropout\n            The dropout rate for every convolutional layer.\n        **kwargs\n            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.\n\n        Inputs\n        ------\n        x of shape `(batch_size, input_chunk_length, input_size)`\n            Tensor containing the features of the input sequence.\n\n        Outputs\n        -------\n        y of shape `(batch_size, input_chunk_length, target_size, nr_params)`\n            Tensor containing the predictions of the next 'output_chunk_length' points in the last\n            'output_chunk_length' entries of the tensor. The entries before contain the data points\n            leading up to the first prediction, all in chronological order.\n        "
        super().__init__(**kwargs)
        self.input_size = input_size
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_length = target_length
        self.target_size = target_size
        self.nr_params = nr_params
        self.dilation_base = dilation_base
        self.dropout = MonteCarloDropout(p=dropout)
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(math.log((self.input_chunk_length - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1, dilation_base))
            logger.info('Number of layers chosen: ' + str(num_layers))
        elif num_layers is None:
            num_layers = math.ceil((self.input_chunk_length - 1) / (kernel_size - 1) / 2)
            logger.info('Number of layers chosen: ' + str(num_layers))
        self.num_layers = num_layers
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(num_filters, kernel_size, dilation_base, self.dropout, weight_norm, i, num_layers, self.input_size, target_size * nr_params)
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    @io_processor
    def forward(self, x_in: Tuple):
        if False:
            for i in range(10):
                print('nop')
        (x, _) = x_in
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        for res_block in self.res_blocks_list:
            x = res_block(x)
        x = x.transpose(1, 2)
        x = x.view(batch_size, self.input_chunk_length, self.target_size, self.nr_params)
        return x

    @property
    def first_prediction_index(self) -> int:
        if False:
            i = 10
            return i + 15
        return -self.output_chunk_length

class TCNModel(PastCovariatesTorchModel):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, kernel_size: int=3, num_filters: int=3, num_layers: Optional[int]=None, dilation_base: int=2, weight_norm: bool=False, dropout: float=0.2, **kwargs):
        if False:
            i = 10
            return i + 15
        'Temporal Convolutional Network Model (TCN).\n\n        This is an implementation of a dilated TCN used for forecasting, inspired from [1]_.\n\n        This model supports past covariates (known for `input_chunk_length` points before prediction time).\n\n        Parameters\n        ----------\n        input_chunk_length\n            Number of past time steps that are fed to the forecasting module.\n        output_chunk_length\n            Number of time steps the torch module will predict into the future at once.\n        kernel_size\n            The size of every kernel in a convolutional layer.\n        num_filters\n            The number of filters in a convolutional layer of the TCN.\n        weight_norm\n            Boolean value indicating whether to use weight normalization.\n        dilation_base\n            The base of the exponent that will determine the dilation on every level.\n        num_layers\n            The number of convolutional layers.\n        dropout\n            The dropout rate for every convolutional layer. This is compatible with Monte Carlo dropout\n            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at\n            prediction time).\n        **kwargs\n            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and\n            Darts\' :class:`TorchForecastingModel`.\n\n        loss_fn\n            PyTorch loss function used for training.\n            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.\n            Default: ``torch.nn.MSELoss()``.\n        likelihood\n            One of Darts\' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for\n            probabilistic forecasts. Default: ``None``.\n        torch_metrics\n            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found\n            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.\n        optimizer_cls\n            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.\n        optimizer_kwargs\n            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{\'lr\': 1e-3}``\n            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``\n            will be used. Default: ``None``.\n        lr_scheduler_cls\n            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds\n            to using a constant learning rate. Default: ``None``.\n        lr_scheduler_kwargs\n            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.\n        use_reversible_instance_norm\n            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.\n            It is only applied to the features of the target series and not the covariates.\n        batch_size\n            Number of time series (input and output sequences) used in each training pass. Default: ``32``.\n        n_epochs\n            Number of epochs over which to train the model. Default: ``100``.\n        model_name\n            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,\n            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part\n            of the name is formatted with the local date and time, while PID is the processed ID (preventing models\n            spawned at the same time by different processes to share the same model_name). E.g.,\n            ``"2021-06-14_09_53_32_torch_model_run_44607"``.\n        work_dir\n            Path of the working directory, where to save checkpoints and Tensorboard summaries.\n            Default: current working directory.\n        log_tensorboard\n            If set, use Tensorboard to log the different parameters. The logs will be located in:\n            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.\n        nr_epochs_val_period\n            Number of epochs to wait before evaluating the validation loss (if a validation\n            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.\n        force_reset\n            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will\n            be discarded). Default: ``False``.\n        save_checkpoints\n            Whether or not to automatically save the untrained model and checkpoints from training.\n            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where\n            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,\n            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using\n            :func:`save()` and loaded using :func:`load()`. Default: ``False``.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        random_state\n            Control the randomness of the weights initialization. Check this\n            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.\n            Default: ``None``.\n        pl_trainer_kwargs\n            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets\n            that performs the training, validation and prediction processes. These presets include automatic\n            checkpointing, tensorboard logging, setting the torch device and more.\n            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer\n            object. Check the `PL Trainer documentation\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the\n            supported kwargs. Default: ``None``.\n            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",\n            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``\n            dict:rgs``\n            dict:\n\n\n            - ``{"accelerator": "cpu"}`` for CPU,\n            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),\n            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.\n\n            For more info, see here:\n            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and\n            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus\n\n            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts\'\n            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.\n            The model will stop training early if the validation loss `val_loss` does not improve beyond\n            specifications. For more information on callbacks, visit:\n            `PyTorch Lightning Callbacks\n            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from pytorch_lightning.callbacks.early_stopping import EarlyStopping\n\n                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over\n                # a period of 5 epochs (`patience`)\n                my_stopper = EarlyStopping(\n                    monitor="val_loss",\n                    patience=5,\n                    min_delta=0.05,\n                    mode=\'min\',\n                )\n\n                pl_trainer_kwargs={"callbacks": [my_stopper]}\n            ..\n\n            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional\n            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.\n        show_warnings\n            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of\n            your forecasting use case. Default: ``False``.\n\n        References\n        ----------\n        .. [1] https://arxiv.org/abs/1803.01271\n        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import TCNModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series[\'p (mbar)\'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series[\'rain (mm)\'][:100]\n        >>> # `output_chunk_length` must be strictly smaller than `input_chunk_length`\n        >>> model = TCNModel(\n        >>>     input_chunk_length=12,\n        >>>     output_chunk_length=6,\n        >>>     n_epochs=20,\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[-80.48476824],\n               [-80.47896667],\n               [-41.77135603],\n               [-41.76158729],\n               [-41.76854107],\n               [-41.78166819]])\n\n        .. note::\n            `DeepTCN example notebook <https://unit8co.github.io/darts/examples/09-DeepTCN-examples.html>`_ presents\n            techniques that can be used to improve the forecasts quality compared to this simple usage example.\n        '
        raise_if_not(kernel_size < input_chunk_length, 'The kernel size must be strictly smaller than the input length.', logger)
        raise_if_not(output_chunk_length < input_chunk_length, 'The output length must be strictly smaller than the input length', logger)
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.weight_norm = weight_norm

    @property
    def supports_multivariate(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        if False:
            print('Hello World!')
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return _TCNModule(input_size=input_dim, target_size=output_dim, nr_params=nr_params, kernel_size=self.kernel_size, num_filters=self.num_filters, num_layers=self.num_layers, dilation_base=self.dilation_base, target_length=self.output_chunk_length, dropout=self.dropout, weight_norm=self.weight_norm, **self.pl_module_params)

    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> PastCovariatesShiftedDataset:
        if False:
            while True:
                i = 10
        return PastCovariatesShiftedDataset(target_series=target, covariates=past_covariates, length=self.input_chunk_length, shift=self.output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=self.uses_static_covariates)