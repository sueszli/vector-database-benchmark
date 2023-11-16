"""
D-Linear
--------
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from darts.logging import raise_if
from darts.models.forecasting.pl_forecasting_module import PLMixedCovariatesModule, io_processor
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
MixedCovariatesTrainTensorType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class _MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if kernel_size % 2 == 0:
            self.padding_size_left = kernel_size // 2 - 1
            self.padding_size_right = kernel_size // 2
        else:
            self.padding_size_left = (kernel_size - 1) // 2
            self.padding_size_right = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if False:
            return 10
        front = x[:, 0:1, :].repeat(1, self.padding_size_left, 1)
        end = x[:, -1:, :].repeat(1, self.padding_size_right, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class _SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        if False:
            return 10
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return (res, moving_mean)

class _DLinearModule(PLMixedCovariatesModule):
    """
    DLinear module
    """

    def __init__(self, input_dim, output_dim, future_cov_dim, static_cov_dim, nr_params, shared_weights, kernel_size, const_init, **kwargs):
        if False:
            print('Hello World!')
        'PyTorch module implementing the DLinear architecture.\n\n        Parameters\n        ----------\n        input_dim\n            The number of input components (target + optional covariate)\n        output_dim\n            Number of output components in the target\n        future_cov_dim\n            Number of components in the future covariates\n        static_cov_dim\n            Dimensionality of the static covariates\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used).\n        shared_weights\n            Whether to use shared weights for the components of the series.\n            ** Ignores covariates when True. **\n        kernel_size\n            The size of the kernel for the moving average\n        const_init\n            Whether to initialize the weights to 1/in_len\n        **kwargs\n            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.\n\n        Inputs\n        ------\n        x of shape `(batch_size, input_chunk_length)`\n            Tensor containing the input sequence.\n\n        Outputs\n        -------\n        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`\n            Tensor containing the output of the NBEATS module.\n        '
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.const_init = const_init
        self.decomposition = _SeriesDecomp(kernel_size)
        self.shared_weights = shared_weights

        def _create_linear_layer(in_dim, out_dim):
            if False:
                for i in range(10):
                    print('nop')
            layer = nn.Linear(in_dim, out_dim)
            if self.const_init:
                layer.weight = nn.Parameter(1.0 / in_dim * torch.ones(layer.weight.shape))
            return layer
        if self.shared_weights:
            layer_in_dim = self.input_chunk_length
            layer_out_dim = self.output_chunk_length * self.nr_params
        else:
            layer_in_dim = self.input_chunk_length * self.input_dim
            layer_out_dim = self.output_chunk_length * self.output_dim * self.nr_params
            layer_in_dim_static_cov = self.output_dim * self.static_cov_dim
        self.linear_seasonal = _create_linear_layer(layer_in_dim, layer_out_dim)
        self.linear_trend = _create_linear_layer(layer_in_dim, layer_out_dim)
        if self.future_cov_dim != 0:
            self.linear_fut_cov = _create_linear_layer(self.future_cov_dim, self.output_dim * self.nr_params)
        if self.static_cov_dim != 0:
            self.linear_static_cov = _create_linear_layer(layer_in_dim_static_cov, layer_out_dim)

    @io_processor
    def forward(self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
        if False:
            while True:
                i = 10
        '\n        x_in\n            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`\n            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`\n        '
        (x, x_future, x_static) = x_in
        (batch, _, _) = x.shape
        if self.shared_weights:
            x = x[:, :, :self.output_dim]
            (res, trend) = self.decomposition(x)
            seasonal_output = self.linear_seasonal(res.permute(0, 2, 1))
            trend_output = self.linear_trend(trend.permute(0, 2, 1))
            x = seasonal_output + trend_output
            x = x.view(batch, self.output_dim, self.output_chunk_length, self.nr_params)
            x = x.permute(0, 2, 1, 3)
        else:
            (res, trend) = self.decomposition(x)
            seasonal_output = self.linear_seasonal(res.view(batch, -1))
            trend_output = self.linear_trend(trend.view(batch, -1))
            seasonal_output = seasonal_output.view(batch, self.output_chunk_length, self.output_dim * self.nr_params)
            trend_output = trend_output.view(batch, self.output_chunk_length, self.output_dim * self.nr_params)
            x = seasonal_output + trend_output
            if self.future_cov_dim != 0:
                x_future = torch.nn.functional.pad(input=x_future, pad=(0, 0, 0, self.output_chunk_length - x_future.shape[1]), mode='constant', value=0)
                fut_cov_output = self.linear_fut_cov(x_future)
                x = x + fut_cov_output.view(batch, self.output_chunk_length, self.output_dim * self.nr_params)
            if self.static_cov_dim != 0:
                static_cov_output = self.linear_static_cov(x_static.reshape(batch, -1))
                x = x + static_cov_output.view(batch, self.output_chunk_length, self.output_dim * self.nr_params)
            x = x.view(batch, self.output_chunk_length, self.output_dim, self.nr_params)
        return x

class DLinearModel(MixedCovariatesTorchModel):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, shared_weights: bool=False, kernel_size: int=25, const_init: bool=True, use_static_covariates: bool=True, **kwargs):
        if False:
            print('Hello World!')
        'An implementation of the DLinear model, as presented in [1]_.\n\n        This implementation is improved by allowing the optional use of past covariates,\n        future covariates and static covariates, and by making the model optionally probabilistic.\n\n        Parameters\n        ----------\n        input_chunk_length\n            The length of the input sequence fed to the model.\n        output_chunk_length\n            The length of the forecast of the model.\n        shared_weights\n            Whether to use shared weights for all components of multivariate series.\n\n            .. warning::\n                When set to True, covariates will be ignored as a 1-to-1 mapping is\n                required between input dimensions and output dimensions.\n            ..\n\n            Default: False.\n\n        kernel_size\n            The size of the kernel for the moving average (default=25). If the size of the kernel is even,\n            the padding will be asymmetrical (shorter on the start/left side).\n        const_init\n            Whether to initialize the weights to 1/in_len. If False, the default PyTorch\n            initialization is used (default=\'True\').\n        use_static_covariates\n            Whether the model should use static covariate information in case the input `series` passed to ``fit()``\n            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce\n            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.\n        **kwargs\n            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and\n            Darts\' :class:`TorchForecastingModel`.\n\n        loss_fn\n            PyTorch loss function used for training.\n            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.\n            Default: ``torch.nn.MSELoss()``.\n        likelihood\n            One of Darts\' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for\n            probabilistic forecasts. Default: ``None``.\n        torch_metrics\n            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found\n            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.\n        optimizer_cls\n            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.\n        optimizer_kwargs\n            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{\'lr\': 1e-3}``\n            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``\n            will be used. Default: ``None``.\n        lr_scheduler_cls\n            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds\n            to using a constant learning rate. Default: ``None``.\n        lr_scheduler_kwargs\n            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.\n        use_reversible_instance_norm\n            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.\n            It is only applied to the features of the target series and not the covariates.\n        batch_size\n            Number of time series (input and output sequences) used in each training pass. Default: ``32``.\n        n_epochs\n            Number of epochs over which to train the model. Default: ``100``.\n        model_name\n            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,\n            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part\n            of the name is formatted with the local date and time, while PID is the processed ID (preventing models\n            spawned at the same time by different processes to share the same model_name). E.g.,\n            ``"2021-06-14_09_53_32_torch_model_run_44607"``.\n        work_dir\n            Path of the working directory, where to save checkpoints and Tensorboard summaries.\n            Default: current working directory.\n        log_tensorboard\n            If set, use Tensorboard to log the different parameters. The logs will be located in:\n            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.\n        nr_epochs_val_period\n            Number of epochs to wait before evaluating the validation loss (if a validation\n            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.\n        force_reset\n            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will\n            be discarded). Default: ``False``.\n        save_checkpoints\n            Whether or not to automatically save the untrained model and checkpoints from training.\n            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where\n            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,\n            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using\n            :func:`save()` and loaded using :func:`load()`. Default: ``False``.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        random_state\n            Control the randomness of the weights initialization. Check this\n            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.\n            Default: ``None``.\n        pl_trainer_kwargs\n            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets\n            that performs the training, validation and prediction processes. These presets include automatic\n            checkpointing, tensorboard logging, setting the torch device and more.\n            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer\n            object. Check the `PL Trainer documentation\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the\n            supported kwargs. Default: ``None``.\n            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",\n            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``\n            dict:\n\n\n            - ``{"accelerator": "cpu"}`` for CPU,\n            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),\n            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.\n\n            For more info, see here:\n            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and\n            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus\n\n            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts\'\n            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.\n            The model will stop training early if the validation loss `val_loss` does not improve beyond\n            specifications. For more information on callbacks, visit:\n            `PyTorch Lightning Callbacks\n            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from pytorch_lightning.callbacks.early_stopping import EarlyStopping\n\n                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over\n                # a period of 5 epochs (`patience`)\n                my_stopper = EarlyStopping(\n                    monitor="val_loss",\n                    patience=5,\n                    min_delta=0.05,\n                    mode=\'min\',\n                )\n\n                pl_trainer_kwargs={"callbacks": [my_stopper]}\n            ..\n\n            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional\n            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.\n        show_warnings\n            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of\n            your forecasting use case. Default: ``False``.\n\n        References\n        ----------\n        .. [1] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022).\n               Are Transformers Effective for Time Series Forecasting?. arXiv preprint arXiv:2205.13504.\n        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import DLinearModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series[\'p (mbar)\'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series[\'rain (mm)\'][:100]\n        >>> # optionally, use future temperatures (pretending this component is a forecast)\n        >>> future_cov = series[\'T (degC)\'][:106]\n        >>> # predict 6 pressure values using the 12 past values of pressure and rainfall, as well as the 6 temperature\n        >>> # values corresponding to the forecasted period\n        >>> model = DLinearModel(\n        >>>     input_chunk_length=6,\n        >>>     output_chunk_length=6,\n        >>>     n_epochs=20,\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[667.20957388],\n               [666.76986848],\n               [666.67733306],\n               [666.06625381],\n               [665.8529289 ],\n               [665.75320573]])\n\n        .. note::\n            This simple usage example produces poor forecasts. In order to obtain better performance, user should\n            transform the input data, increase the number of epochs, use a validation set, optimize the hyper-\n            parameters, ...\n        '
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        self.shared_weights = shared_weights
        self.kernel_size = kernel_size
        self.const_init = const_init
        self._considers_static_covariates = use_static_covariates

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> torch.nn.Module:
        if False:
            for i in range(10):
                print('nop')
        raise_if(self.shared_weights and (train_sample[1] is not None or train_sample[2] is not None), 'Covariates have been provided, but the model has been built with shared_weights=True.' + 'Please set shared_weights=False to use covariates.')
        input_dim = train_sample[0].shape[1] + sum((train_sample[i].shape[1] if train_sample[i] is not None else 0 for i in (1, 2)))
        future_cov_dim = train_sample[3].shape[1] if train_sample[3] is not None else 0
        static_cov_dim = train_sample[4].shape[1] if train_sample[4] is not None else 0
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return _DLinearModule(input_dim=input_dim, output_dim=output_dim, future_cov_dim=future_cov_dim, static_cov_dim=static_cov_dim, nr_params=nr_params, shared_weights=self.shared_weights, kernel_size=self.kernel_size, const_init=self.const_init, **self.pl_module_params)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            print('Hello World!')
        return True

    @property
    def supports_static_covariates(self) -> bool:
        if False:
            return 10
        return True

    @property
    def supports_future_covariates(self) -> bool:
        if False:
            return 10
        return not self.shared_weights

    @property
    def supports_past_covariates(self) -> bool:
        if False:
            print('Hello World!')
        return not self.shared_weights