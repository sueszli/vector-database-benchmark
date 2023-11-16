"""
Transformer Model
-----------------
"""
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.components import glu_variants, layer_norm_variants
from darts.models.components.glu_variants import GLU_FFN
from darts.models.components.transformer import CustomFeedForwardDecoderLayer, CustomFeedForwardEncoderLayer
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule, io_processor
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
logger = get_logger(__name__)
BUILT_IN = ['relu', 'gelu']
FFN = GLU_FFN + BUILT_IN

def _generate_coder(d_model, dim_ff, dropout, nhead, num_layers, norm_layer, coder_cls, layer_cls, ffn_cls):
    if False:
        i = 10
        return i + 15
    "Generates an Encoder or Decoder with one of Darts' Feed-forward Network variants.\n    Parameters\n    ----------\n    coder_cls\n        Either `torch.nn.TransformerEncoder` or `...TransformerDecoder`\n    layer_cls\n        Either `darts.models.components.transformer.CustomFeedForwardEncoderLayer`,\n        `...CustomFeedForwardDecoderLayer`, `nn.TransformerEncoderLayer`, or `nn.TransformerDecoderLayer`.\n    ffn_cls\n        One of Darts' Position-wise Feed-Forward Network variants `from darts.models.components.glu_variants`\n    "
    ffn = dict(ffn=ffn_cls(d_model=d_model, d_ff=dim_ff, dropout=dropout)) if ffn_cls else dict()
    layer = layer_cls(**ffn, dropout=dropout, d_model=d_model, nhead=nhead, dim_feedforward=dim_ff)
    return coder_cls(layer, num_layers=num_layers, norm=norm_layer(d_model))

class _PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        if False:
            while True:
                i = 10
        'An implementation of positional encoding as described in \'Attention is All you Need\' by Vaswani et al. (2017)\n\n        Parameters\n        ----------\n        d_model\n            The number of expected features in the transformer encoder/decoder inputs.\n            Last dimension of the input.\n        dropout\n            Fraction of neurons affected by Dropout (default=0.1).\n        max_len\n            The dimensionality of the computed positional encoding array.\n            Only its first "input_size" elements will be considered in the output.\n\n        Inputs\n        ------\n        x of shape `(batch_size, input_size, d_model)`\n            Tensor containing the embedded time series.\n\n        Outputs\n        -------\n        y of shape `(batch_size, input_size, d_model)`\n            Tensor containing the embedded time series enhanced with positional encoding.\n        '
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class _TransformerModule(PLPastCovariatesModule):

    def __init__(self, input_size: int, output_size: int, nr_params: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float, activation: str, norm_type: Union[str, nn.Module, None]=None, custom_encoder: Optional[nn.Module]=None, custom_decoder: Optional[nn.Module]=None, **kwargs):
        if False:
            return 10
        'PyTorch module implementing a Transformer to be used in `TransformerModel`.\n\n        PyTorch module implementing a simple encoder-decoder transformer architecture.\n\n        Parameters\n        ----------\n        input_size\n            The dimensionality of the TimeSeries instances that will be fed to the the fit and predict functions.\n        output_size\n            The dimensionality of the output time series.\n        nr_params\n            The number of parameters of the likelihood (or 1 if no likelihood is used).\n        d_model\n            The number of expected features in the transformer encoder/decoder inputs.\n        nhead\n            The number of heads in the multiheadattention model.\n        num_encoder_layers\n            The number of encoder layers in the encoder.\n        num_decoder_layers\n            The number of decoder layers in the decoder.\n        dim_feedforward\n            The dimension of the feedforward network model.\n        dropout\n            Fraction of neurons affected by Dropout.\n        activation\n            The activation function of encoder/decoder intermediate layer.\n        norm_type: str | nn.Module | None\n            The type of LayerNorm variant to use.\n        custom_encoder\n            A custom transformer encoder provided by the user (default=None).\n        custom_decoder\n            A custom transformer decoder provided by the user (default=None).\n        **kwargs\n            All parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.\n\n        Inputs\n        ------\n        x of shape `(batch_size, input_chunk_length, input_size)`\n            Tensor containing the features of the input sequence.\n\n        Outputs\n        -------\n        y of shape `(batch_size, output_chunk_length, target_size, nr_params)`\n            Tensor containing the prediction at the last time step of the sequence.\n        '
        super().__init__(**kwargs)
        self.input_size = input_size
        self.target_size = output_size
        self.nr_params = nr_params
        self.target_length = self.output_chunk_length
        self.encoder = nn.Linear(input_size, d_model)
        self.positional_encoding = _PositionalEncoding(d_model, dropout, self.input_chunk_length)
        if isinstance(norm_type, str):
            try:
                self.layer_norm = getattr(layer_norm_variants, norm_type)
            except AttributeError:
                raise_log(AttributeError('please provide a valid layer norm type'))
        else:
            self.layer_norm = norm_type
        raise_if_not(activation in FFN, f"'{activation}' is not in {FFN}")
        if activation in GLU_FFN:
            raise_if(custom_encoder is not None or custom_decoder is not None, f'Cannot use `custom_encoder` or `custom_decoder` along with an `activation` from {GLU_FFN}', logger=logger)
            ffn_cls = getattr(glu_variants, activation)
            activation = None
            custom_encoder = _generate_coder(d_model, dim_feedforward, dropout, nhead, num_encoder_layers, self.layer_norm if self.layer_norm else nn.LayerNorm, coder_cls=nn.TransformerEncoder, layer_cls=CustomFeedForwardEncoderLayer, ffn_cls=ffn_cls)
            custom_decoder = _generate_coder(d_model, dim_feedforward, dropout, nhead, num_decoder_layers, self.layer_norm if self.layer_norm else nn.LayerNorm, coder_cls=nn.TransformerDecoder, layer_cls=CustomFeedForwardDecoderLayer, ffn_cls=ffn_cls)
        if self.layer_norm and custom_decoder is None:
            custom_encoder = _generate_coder(d_model, dim_feedforward, dropout, nhead, num_encoder_layers, self.layer_norm, coder_cls=nn.TransformerEncoder, layer_cls=nn.TransformerEncoderLayer, ffn_cls=None)
            custom_decoder = _generate_coder(d_model, dim_feedforward, dropout, nhead, num_decoder_layers, self.layer_norm, coder_cls=nn.TransformerDecoder, layer_cls=nn.TransformerDecoderLayer, ffn_cls=None)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, custom_encoder=custom_encoder, custom_decoder=custom_decoder)
        self.decoder = nn.Linear(d_model, self.target_length * self.target_size * self.nr_params)

    def _create_transformer_inputs(self, data):
        if False:
            while True:
                i = 10
        src = data.permute(1, 0, 2)
        tgt = src[-1:, :, :]
        return (src, tgt)

    @io_processor
    def forward(self, x_in: Tuple):
        if False:
            return 10
        (data, _) = x_in
        (src, tgt) = self._create_transformer_inputs(data)
        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.positional_encoding(src)
        tgt = self.encoder(tgt) * math.sqrt(self.input_size)
        tgt = self.positional_encoding(tgt)
        x = self.transformer(src=src, tgt=tgt)
        out = self.decoder(x)
        predictions = out[0, :, :]
        predictions = predictions.view(-1, self.target_length, self.target_size, self.nr_params)
        return predictions

class TransformerModel(PastCovariatesTorchModel):

    def __init__(self, input_chunk_length: int, output_chunk_length: int, d_model: int=64, nhead: int=4, num_encoder_layers: int=3, num_decoder_layers: int=3, dim_feedforward: int=512, dropout: float=0.1, activation: str='relu', norm_type: Union[str, nn.Module, None]=None, custom_encoder: Optional[nn.Module]=None, custom_decoder: Optional[nn.Module]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Transformer model\n\n        Transformer is a state-of-the-art deep learning model introduced in 2017. It is an encoder-decoder\n        architecture whose core feature is the \'multi-head attention\' mechanism, which is able to\n        draw intra-dependencies within the input vector and within the output vector (\'self-attention\')\n        as well as inter-dependencies between input and output vectors (\'encoder-decoder attention\').\n        The multi-head attention mechanism is highly parallelizable, which makes the transformer architecture\n        very suitable to be trained with GPUs.\n\n        The transformer architecture implemented here is based on [1]_.\n\n        This model supports past covariates (known for `input_chunk_length` points before prediction time).\n\n        Parameters\n        ----------\n        input_chunk_length\n            Number of time steps to be input to the forecasting module.\n        output_chunk_length\n            Number of time steps to be output by the forecasting module.\n        d_model\n            The number of expected features in the transformer encoder/decoder inputs (default=64).\n        nhead\n            The number of heads in the multi-head attention mechanism (default=4).\n        num_encoder_layers\n            The number of encoder layers in the encoder (default=3).\n        num_decoder_layers\n            The number of decoder layers in the decoder (default=3).\n        dim_feedforward\n            The dimension of the feedforward network model (default=512).\n        dropout\n            Fraction of neurons affected by Dropout (default=0.1).\n        activation\n            The activation function of encoder/decoder intermediate layer, (default=\'relu\').\n            can be one of the glu variant\'s FeedForward Network (FFN)[2]. A feedforward network is a\n            fully-connected layer with an activation. The glu variant\'s FeedForward Network are a series\n            of FFNs designed to work better with Transformer based models. ["GLU", "Bilinear", "ReGLU", "GEGLU",\n            "SwiGLU", "ReLU", "GELU"] or one the pytorch internal activations ["relu", "gelu"]\n        norm_type: str | nn.Module\n            The type of LayerNorm variant to use.  Default: ``None``. Available options are\n            ["LayerNorm", "RMSNorm", "LayerNormNoBias"], or provide a custom nn.Module.\n        custom_encoder\n            A custom user-provided encoder module for the transformer (default=None).\n        custom_decoder\n            A custom user-provided decoder module for the transformer (default=None).\n        **kwargs\n            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and\n            Darts\' :class:`TorchForecastingModel`.\n\n        loss_fn\n            PyTorch loss function used for training.\n            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.\n            Default: ``torch.nn.MSELoss()``.\n        likelihood\n            One of Darts\' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for\n            probabilistic forecasts. Default: ``None``.\n        torch_metrics\n            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found\n            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.\n        optimizer_cls\n            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.\n        optimizer_kwargs\n            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{\'lr\': 1e-3}``\n            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``\n            will be used. Default: ``None``.\n        lr_scheduler_cls\n            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds\n            to using a constant learning rate. Default: ``None``.\n        lr_scheduler_kwargs\n            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.\n        use_reversible_instance_norm\n            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [3]_.\n            It is only applied to the features of the target series and not the covariates.\n        batch_size\n            Number of time series (input and output sequences) used in each training pass. Default: ``32``.\n        n_epochs\n            Number of epochs over which to train the model. Default: ``100``.\n        model_name\n            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,\n            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part\n            of the name is formatted with the local date and time, while PID is the processed ID (preventing models\n            spawned at the same time by different processes to share the same model_name). E.g.,\n            ``"2021-06-14_09_53_32_torch_model_run_44607"``.\n        work_dir\n            Path of the working directory, where to save checkpoints and Tensorboard summaries.\n            Default: current working directory.\n        log_tensorboard\n            If set, use Tensorboard to log the different parameters. The logs will be located in:\n            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.\n        nr_epochs_val_period\n            Number of epochs to wait before evaluating the validation loss (if a validation\n            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.\n        force_reset\n            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will\n            be discarded). Default: ``False``.\n        save_checkpoints\n            Whether or not to automatically save the untrained model and checkpoints from training.\n            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where\n            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,\n            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using\n            :func:`save()` and loaded using :func:`load()`. Default: ``False``.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        random_state\n            Control the randomness of the weights initialization. Check this\n            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.\n            Default: ``None``.\n        pl_trainer_kwargs\n            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets\n            that performs the training, validation and prediction processes. These presets include automatic\n            checkpointing, tensorboard logging, setting the torch device and more.\n            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer\n            object. Check the `PL Trainer documentation\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the\n            supported kwargs. Default: ``None``.\n            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",\n            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``\n            dict:\n\n\n            - ``{"accelerator": "cpu"}`` for CPU,\n            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),\n            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.\n\n            For more info, see here:\n            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and\n            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus\n\n            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts\'\n            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.\n            The model will stop training early if the validation loss `val_loss` does not improve beyond\n            specifications. For more information on callbacks, visit:\n            `PyTorch Lightning Callbacks\n            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from pytorch_lightning.callbacks.early_stopping import EarlyStopping\n\n                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over\n                # a period of 5 epochs (`patience`)\n                my_stopper = EarlyStopping(\n                    monitor="val_loss",\n                    patience=5,\n                    min_delta=0.05,\n                    mode=\'min\',\n                )\n\n                pl_trainer_kwargs={"callbacks": [my_stopper]}\n            ..\n\n            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional\n            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.\n        show_warnings\n            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of\n            your forecasting use case. Default: ``False``.\n\n        References\n        ----------\n        .. [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,\n        and Illia Polosukhin, "Attention Is All You Need", 2017. In Advances in Neural Information Processing Systems,\n        pages 6000-6010. https://arxiv.org/abs/1706.03762.\n        .. [2] Shazeer, Noam, "GLU Variants Improve Transformer", 2020. arVix https://arxiv.org/abs/2002.05202.\n        .. [3] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p\n\n        Notes\n        -----\n        Disclaimer:\n        This current implementation is fully functional and can already produce some good predictions. However,\n        it is still limited in how it uses the Transformer architecture because the `tgt` input of\n        `torch.nn.Transformer` is not utilized to its full extent. Currently, we simply pass the last value of the\n        `src` input to `tgt`. To get closer to the way the Transformer is usually used in language models, we\n        should allow the model to consume its own output as part of the `tgt` argument, such that when predicting\n        sequences of values, the input to the `tgt` argument would grow as outputs of the transformer model would be\n        added to it. Of course, the training of the model would have to be adapted accordingly.\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import TransformerModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series[\'p (mbar)\'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series[\'rain (mm)\'][:100]\n        >>> model = TransformerModel(\n        >>>     input_chunk_length=6,\n        >>>     output_chunk_length=6,\n        >>>     n_epochs=20\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[5.40498034],\n               [5.36561899],\n               [5.80616883],\n               [6.48695488],\n               [7.63158655],\n               [5.65417736]])\n\n        .. note::\n            `Transformer example notebook <https://unit8co.github.io/darts/examples/06-Transformer-examples.html>`_\n            presents techniques that can be used to improve the forecasts quality compared to this simple usage\n            example.\n        '
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_type = norm_type
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder

    @property
    def supports_multivariate(self) -> bool:
        if False:
            return 10
        return True

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        if False:
            print('Hello World!')
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return _TransformerModule(input_size=input_dim, output_size=output_dim, nr_params=nr_params, d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers, num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation, norm_type=self.norm_type, custom_encoder=self.custom_encoder, custom_decoder=self.custom_decoder, **self.pl_module_params)