"""
Time-series Dense Encoder (TiDE)
------
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLMixedCovariatesModule, io_processor
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
MixedCovariatesTrainTensorType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
logger = get_logger(__name__)

class _ResidualBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout: float, use_layer_norm: bool):
        if False:
            i = 10
            return i + 15
        'Pytorch module implementing the Residual Block from the TiDE paper.'
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_dim), nn.Dropout(dropout))
        self.skip = nn.Linear(input_dim, output_dim)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        x = self.dense(x) + self.skip(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x

class _TideModule(PLMixedCovariatesModule):

    def __init__(self, input_dim: int, output_dim: int, future_cov_dim: int, static_cov_dim: int, nr_params: int, num_encoder_layers: int, num_decoder_layers: int, decoder_output_dim: int, hidden_size: int, temporal_decoder_hidden: int, temporal_width_past: int, temporal_width_future: int, use_layer_norm: bool, dropout: float, **kwargs):
        if False:
            print('Hello World!')
        'Pytorch module implementing the TiDE architecture.\n\n        Parameters\n        ----------\n        input_dim\n            The number of input components (target + optional past covariates + optional future covariates).\n        output_dim\n            Number of output components in the target.\n        future_cov_dim\n            Number of future covariates.\n        static_cov_dim\n            Number of static covariates.\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used).\n        num_encoder_layers\n            Number of stacked Residual Blocks in the encoder.\n        num_decoder_layers\n            Number of stacked Residual Blocks in the decoder.\n        decoder_output_dim\n            The number of output components of the decoder.\n        hidden_size\n            The width of the hidden layers in the encoder/decoder Residual Blocks.\n        temporal_decoder_hidden\n            The width of the hidden layers in the temporal decoder.\n        temporal_width_past\n            The width of the past covariate embedding space.\n        temporal_width_future\n            The width of the future covariate embedding space.\n        use_layer_norm\n            Whether to use layer normalization in the Residual Blocks.\n        dropout\n            Dropout probability\n        **kwargs\n            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.\n\n        Inputs\n        ------\n        x\n            Tuple of Tensors `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and\n            `x_future`is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`\n        Outputs\n        -------\n        y\n            Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`\n\n        '
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = input_dim - output_dim - future_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.past_cov_projection = None
        if self.past_cov_dim and temporal_width_past:
            self.past_cov_projection = _ResidualBlock(input_dim=self.past_cov_dim, output_dim=temporal_width_past, hidden_size=hidden_size, use_layer_norm=use_layer_norm, dropout=dropout)
            past_covariates_flat_dim = self.input_chunk_length * temporal_width_past
        elif self.past_cov_dim:
            past_covariates_flat_dim = self.input_chunk_length * self.past_cov_dim
        else:
            past_covariates_flat_dim = 0
        self.future_cov_projection = None
        if future_cov_dim and self.temporal_width_future:
            self.future_cov_projection = _ResidualBlock(input_dim=future_cov_dim, output_dim=temporal_width_future, hidden_size=hidden_size, use_layer_norm=use_layer_norm, dropout=dropout)
            historical_future_covariates_flat_dim = (self.input_chunk_length + self.output_chunk_length) * temporal_width_future
        elif future_cov_dim:
            historical_future_covariates_flat_dim = (self.input_chunk_length + self.output_chunk_length) * future_cov_dim
        else:
            historical_future_covariates_flat_dim = 0
        encoder_dim = self.input_chunk_length * output_dim + past_covariates_flat_dim + historical_future_covariates_flat_dim + static_cov_dim
        self.encoders = nn.Sequential(_ResidualBlock(input_dim=encoder_dim, output_dim=hidden_size, hidden_size=hidden_size, use_layer_norm=use_layer_norm, dropout=dropout), *[_ResidualBlock(input_dim=hidden_size, output_dim=hidden_size, hidden_size=hidden_size, use_layer_norm=use_layer_norm, dropout=dropout) for _ in range(num_encoder_layers - 1)])
        self.decoders = nn.Sequential(*[_ResidualBlock(input_dim=hidden_size, output_dim=hidden_size, hidden_size=hidden_size, use_layer_norm=use_layer_norm, dropout=dropout) for _ in range(num_decoder_layers - 1)], _ResidualBlock(input_dim=hidden_size, output_dim=decoder_output_dim * self.output_chunk_length * self.nr_params, hidden_size=hidden_size, use_layer_norm=use_layer_norm, dropout=dropout))
        decoder_input_dim = decoder_output_dim * self.nr_params
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim
        self.temporal_decoder = _ResidualBlock(input_dim=decoder_input_dim, output_dim=output_dim * self.nr_params, hidden_size=temporal_decoder_hidden, use_layer_norm=use_layer_norm, dropout=dropout)
        self.lookback_skip = nn.Linear(self.input_chunk_length, self.output_chunk_length * self.nr_params)

    @io_processor
    def forward(self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]) -> torch.Tensor:
        if False:
            while True:
                i = 10
        'TiDE model forward pass.\n        Parameters\n        ----------\n        x_in\n            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`\n            is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`\n        Returns\n        -------\n        torch.Tensor\n            The output Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`\n        '
        (x, x_future_covariates, x_static_covariates) = x_in
        x_lookback = x[:, :, :self.output_dim]
        if self.future_cov_dim:
            x_dynamic_future_covariates = torch.cat([x[:, :, None if self.future_cov_dim == 0 else -self.future_cov_dim:], x_future_covariates], dim=1)
            if self.temporal_width_future:
                x_dynamic_future_covariates = self.future_cov_projection(x_dynamic_future_covariates)
        else:
            x_dynamic_future_covariates = None
        if self.past_cov_dim:
            x_dynamic_past_covariates = x[:, :, self.output_dim:self.output_dim + self.past_cov_dim]
            if self.temporal_width_past:
                x_dynamic_past_covariates = self.past_cov_projection(x_dynamic_past_covariates)
        else:
            x_dynamic_past_covariates = None
        encoded = [x_lookback, x_dynamic_past_covariates, x_dynamic_future_covariates, x_static_covariates]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)
        temporal_decoder_input = [decoded, x_dynamic_future_covariates[:, -self.output_chunk_length:, :] if self.future_cov_dim > 0 else None]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]
        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)
        y = temporal_decoded + skip.reshape_as(temporal_decoded)
        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        return y

class TiDEModel(MixedCovariatesTorchModel):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, num_encoder_layers: int=1, num_decoder_layers: int=1, decoder_output_dim: int=16, hidden_size: int=128, temporal_width_past: int=4, temporal_width_future: int=4, temporal_decoder_hidden: int=32, use_layer_norm: bool=False, dropout: float=0.1, use_static_covariates: bool=True, **kwargs):
        if False:
            i = 10
            return i + 15
        'An implementation of the TiDE model, as presented in [1]_.\n\n        TiDE is similar to Transformers (implemented in :class:`TransformerModel`),\n        but attempts to provide better performance at lower computational cost by introducing\n        multilayer perceptron (MLP)-based encoder-decoders without attention.\n\n        The model is implemented as a :class:`MixedCovariatesTorchModel`, which means that it supports\n        both past and future covariates, as well as static covariates. Probabilistic forecasting is supported through\n        the use of a `likelihood` instead of a `loss_fn`.\n        The original paper does not describe how past covariates are treated in detail, so we assume that they are\n        passed to the encoder as-is.\n\n        The encoder and decoder are implemented as a series of residual blocks. The number of residual blocks in\n        the encoder and decoder can be controlled via ``num_encoder_layers`` and ``num_decoder_layers`` respectively.\n        The width of the layers in the residual blocks can be controlled via ``hidden_size``. Similarly, the width\n        of the layers in the temporal decoder can be controlled via ``temporal_decoder_hidden``.\n\n        Parameters\n        ----------\n        input_chunk_length\n            The length of the input sequence fed to the model.\n        output_chunk_length\n            The length of the forecast of the model.\n        num_encoder_layers\n            The number of residual blocks in the encoder.\n        num_decoder_layers\n            The number of residual blocks in the decoder.\n        decoder_output_dim\n            The dimensionality of the output of the decoder.\n        hidden_size\n            The width of the layers in the residual blocks of the encoder and decoder.\n        temporal_width_past\n            The width of the layers in the past covariate projection residual block. If `0`,\n            will bypass feature projection and use the raw feature data.\n        temporal_width_future\n            The width of the layers in the future covariate projection residual block. If `0`,\n            will bypass feature projection and use the raw feature data.\n        temporal_decoder_hidden\n            The width of the layers in the temporal decoder.\n        use_layer_norm\n            Whether to use layer normalization in the residual blocks.\n        dropout\n            The dropout probability to be used in fully connected layers. This is compatible with Monte Carlo dropout\n            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at\n            prediction time).\n        **kwargs\n            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and\n            Darts\' :class:`TorchForecastingModel`.\n\n        loss_fn\n            PyTorch loss function used for training.\n            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.\n            Default: ``torch.nn.MSELoss()``.\n        likelihood\n            One of Darts\' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for\n            probabilistic forecasts. Default: ``None``.\n        torch_metrics\n            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found\n            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.\n        optimizer_cls\n            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.\n        optimizer_kwargs\n            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{\'lr\': 1e-3}``\n            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``\n            will be used. Default: ``None``.\n        lr_scheduler_cls\n            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds\n            to using a constant learning rate. Default: ``None``.\n        lr_scheduler_kwargs\n            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.\n        use_reversible_instance_norm\n            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.\n            It is only applied to the features of the target series and not the covariates.\n        batch_size\n            Number of time series (input and output sequences) used in each training pass. Default: ``32``.\n        n_epochs\n            Number of epochs over which to train the model. Default: ``100``.\n        model_name\n            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,\n            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part\n            of the name is formatted with the local date and time, while PID is the processed ID (preventing models\n            spawned at the same time by different processes to share the same model_name). E.g.,\n            ``"2021-06-14_09_53_32_torch_model_run_44607"``.\n        work_dir\n            Path of the working directory, where to save checkpoints and Tensorboard summaries.\n            Default: current working directory.\n        log_tensorboard\n            If set, use Tensorboard to log the different parameters. The logs will be located in:\n            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.\n        nr_epochs_val_period\n            Number of epochs to wait before evaluating the validation loss (if a validation\n            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.\n        force_reset\n            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will\n            be discarded). Default: ``False``.\n        save_checkpoints\n            Whether or not to automatically save the untrained model and checkpoints from training.\n            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where\n            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,\n            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using\n            :func:`save()` and loaded using :func:`load()`. Default: ``False``.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        random_state\n            Control the randomness of the weights initialization. Check this\n            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.\n            Default: ``None``.\n        pl_trainer_kwargs\n            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets\n            that performs the training, validation and prediction processes. These presets include automatic\n            checkpointing, tensorboard logging, setting the torch device and more.\n            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer\n            object. Check the `PL Trainer documentation\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the\n            supported kwargs. Default: ``None``.\n            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",\n            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``\n            dict:\n\n\n            - ``{"accelerator": "cpu"}`` for CPU,\n            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),\n            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.\n\n            For more info, see here:\n            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and\n            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus\n\n            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts\'\n            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.\n            The model will stop training early if the validation loss `val_loss` does not improve beyond\n            specifications. For more information on callbacks, visit:\n            `PyTorch Lightning Callbacks\n            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from pytorch_lightning.callbacks.early_stopping import EarlyStopping\n\n                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over\n                # a period of 5 epochs (`patience`)\n                my_stopper = EarlyStopping(\n                    monitor="val_loss",\n                    patience=5,\n                    min_delta=0.05,\n                    mode=\'min\',\n                )\n\n                pl_trainer_kwargs={"callbacks": [my_stopper]}\n            ..\n\n            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional\n            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.\n        show_warnings\n            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of\n            your forecasting use case. Default: ``False``.\n\n        References\n        ----------\n        .. [1] A. Das et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder",\n                http://arxiv.org/abs/2304.08424\n        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import TiDEModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series[\'p (mbar)\'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series[\'rain (mm)\'][:100]\n        >>> # optionally, use future temperatures (pretending this component is a forecast)\n        >>> future_cov = series[\'T (degC)\'][:106]\n        >>> model = TiDEModel(\n        >>>     input_chunk_length=6,\n        >>>     output_chunk_length=6,\n        >>>     n_epochs=20\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[1008.1667634 ],\n               [ 997.08337201],\n               [1017.72035839],\n               [1005.10790392],\n               [ 998.90537286],\n               [1005.91534452]])\n\n        .. note::\n            `TiDE example notebook <https://unit8co.github.io/darts/examples/18-TiDE-examples.html>`_ presents\n            techniques that can be used to improve the forecasts quality compared to this simple usage example.\n        '
        if temporal_width_past < 0 or temporal_width_future < 0:
            raise_log(ValueError('`temporal_width_past` and `temporal_width_future` must be >= 0.'), logger=logger)
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self._considers_static_covariates = use_static_covariates
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> torch.nn.Module:
        if False:
            print('Hello World!')
        (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates, future_target) = train_sample
        input_dim = past_target.shape[1] + (past_covariates.shape[1] if past_covariates is not None else 0) + (historic_future_covariates.shape[1] if historic_future_covariates is not None else 0)
        output_dim = future_target.shape[1]
        future_cov_dim = future_covariates.shape[1] if future_covariates is not None else 0
        static_cov_dim = static_covariates.shape[0] * static_covariates.shape[1] if static_covariates is not None else 0
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        past_cov_dim = input_dim - output_dim - future_cov_dim
        if past_cov_dim and self.temporal_width_past >= past_cov_dim:
            logger.warning(f'number of `past_covariates` features is <= `temporal_width_past`, leading to feature expansion.number of covariates: {past_cov_dim}, `temporal_width_past={self.temporal_width_past}`.')
        if future_cov_dim and self.temporal_width_future >= future_cov_dim:
            logger.warning(f'number of `future_covariates` features is <= `temporal_width_future`, leading to feature expansion.number of covariates: {future_cov_dim}, `temporal_width_future={self.temporal_width_future}`.')
        return _TideModule(input_dim=input_dim, output_dim=output_dim, future_cov_dim=future_cov_dim, static_cov_dim=static_cov_dim, nr_params=nr_params, num_encoder_layers=self.num_encoder_layers, num_decoder_layers=self.num_decoder_layers, decoder_output_dim=self.decoder_output_dim, hidden_size=self.hidden_size, temporal_width_past=self.temporal_width_past, temporal_width_future=self.temporal_width_future, temporal_decoder_hidden=self.temporal_decoder_hidden, use_layer_norm=self.use_layer_norm, dropout=self.dropout, **self.pl_module_params)

    @property
    def supports_static_covariates(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @property
    def supports_multivariate(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def predict(self, n, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if n >= self.output_chunk_length:
            return super().predict(n, *args, **kwargs)
        else:
            return super().predict(self.output_chunk_length, *args, **kwargs)[:n]