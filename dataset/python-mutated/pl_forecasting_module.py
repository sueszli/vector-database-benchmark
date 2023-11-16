"""
This file contains abstract classes for deterministic and probabilistic PyTorch Lightning Modules
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from joblib import Parallel, delayed
from darts.logging import get_logger, raise_if, raise_log
from darts.models.components.layer_norm_variants import RINorm
from darts.timeseries import TimeSeries
from darts.utils.likelihood_models import Likelihood
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.torch import MonteCarloDropout
logger = get_logger(__name__)
tokens = pl.__version__.split('.')
pl_160_or_above = int(tokens[0]) > 1 or (int(tokens[0]) == 1 and int(tokens[1]) >= 6)

def io_processor(forward):
    if False:
        for i in range(10):
            print('nop')
    "Applies some input / output processing to PLForecastingModule.forward.\n    Note that this wrapper must be added to each of PLForecastinModule's subclasses forward methods.\n    Here is an example how to add the decorator:\n\n    ```python\n        @io_processor\n        def forward(self, *args, **kwargs)\n            pass\n    ```\n\n    Applies\n    -------\n    Reversible Instance Normalization\n        normalizes batch input target features, and inverse transform the forward output back to the original scale\n    "

    def forward_wrapper(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not self.use_reversible_instance_norm:
            return forward(self, *args, **kwargs)
        x: Tuple = args[0][0]
        x[:, :, :self.n_targets] = self.rin(x[:, :, :self.n_targets])
        out = forward(self, *((x, *args[0][1:]), *args[1:]), **kwargs)
        if isinstance(out, tuple):
            return (self.rin.inverse(out[0]), *out[1:])
        else:
            return self.rin.inverse(out)
    return forward_wrapper

class PLForecastingModule(pl.LightningModule, ABC):

    @abstractmethod
    def __init__(self, input_chunk_length: int, output_chunk_length: int, train_sample_shape: Optional[Tuple]=None, loss_fn: nn.modules.loss._Loss=nn.MSELoss(), torch_metrics: Optional[Union[torchmetrics.Metric, torchmetrics.MetricCollection]]=None, likelihood: Optional[Likelihood]=None, optimizer_cls: torch.optim.Optimizer=torch.optim.Adam, optimizer_kwargs: Optional[Dict]=None, lr_scheduler_cls: Optional[torch.optim.lr_scheduler._LRScheduler]=None, lr_scheduler_kwargs: Optional[Dict]=None, use_reversible_instance_norm: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        PyTorch Lightning-based Forecasting Module.\n\n        This class is meant to be inherited to create a new PyTorch Lightning-based forecasting module.\n        When subclassing this class, please make sure to add the following methods with the given signatures:\n            - :func:`PLTorchForecastingModel.__init__()`\n            - :func:`PLTorchForecastingModel.forward()`\n            - :func:`PLTorchForecastingModel._produce_train_output()`\n            - :func:`PLTorchForecastingModel._get_batch_prediction()`\n\n        In subclass `MyModel`\'s :func:`__init__` function call ``super(MyModel, self).__init__(**kwargs)`` where\n        ``kwargs`` are the parameters of :class:`PLTorchForecastingModel`.\n\n        Parameters\n        ----------\n        input_chunk_length\n            Number of input past time steps per chunk.\n        output_chunk_length\n            Number of output time steps per chunk.\n        train_sample_shape\n            Shape of the model\'s input, used to instantiate model without calling ``fit_from_dataset`` and\n            perform sanity check on new training/inference datasets used for re-training or prediction.\n        loss_fn\n            PyTorch loss function used for training.\n            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.\n            Default: ``torch.nn.MSELoss()``.\n        torch_metrics\n            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found\n            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.\n        likelihood\n            One of Darts\' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for\n            probabilistic forecasts. Default: ``None``.\n        optimizer_cls\n            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.\n        optimizer_kwargs\n            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{\'lr\': 1e-3}``\n            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``\n            will be used. Default: ``None``.\n        lr_scheduler_cls\n            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds\n            to using a constant learning rate. Default: ``None``.\n        lr_scheduler_kwargs\n            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.\n        use_reversible_instance_norm\n            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [1]_.\n            It is only applied to the features of the target series and not the covariates.\n\n        References\n        ----------\n        .. [1] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p\n        '
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'torch_metrics'])
        raise_if(input_chunk_length is None or output_chunk_length is None, 'Both `input_chunk_length` and `output_chunk_length` must be passed to `PLForecastingModule`', logger)
        self.input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self.criterion = loss_fn
        self.likelihood = likelihood
        self.train_sample_shape = train_sample_shape
        self.n_targets = train_sample_shape[0][1] if train_sample_shape is not None else 1
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        torch_metrics = self.configure_torch_metrics(torch_metrics)
        self.train_metrics = torch_metrics.clone(prefix='train_')
        self.val_metrics = torch_metrics.clone(prefix='val_')
        self.use_reversible_instance_norm = use_reversible_instance_norm
        if use_reversible_instance_norm:
            self.rin = RINorm(input_dim=self.n_targets)
        else:
            self.rin = None
        self.pred_n: Optional[int] = None
        self.pred_num_samples: Optional[int] = None
        self.pred_roll_size: Optional[int] = None
        self.pred_batch_size: Optional[int] = None
        self.pred_n_jobs: Optional[int] = None
        self.predict_likelihood_parameters: Optional[bool] = None

    @property
    def first_prediction_index(self) -> int:
        if False:
            return 10
        '\n        Returns the index of the first predicted within the output of self.model.\n        '
        return 0

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        if False:
            while True:
                i = 10
        super().forward(*args, **kwargs)

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        'performs the training step'
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[-1]
        loss = self._compute_loss(output, target)
        self.log('train_loss', loss, batch_size=train_batch[0].shape[0], prog_bar=True, sync_dist=True)
        self._calculate_metrics(output, target, self.train_metrics)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        'performs the validation step'
        output = self._produce_train_output(val_batch[:-1])
        target = val_batch[-1]
        loss = self._compute_loss(output, target)
        self.log('val_loss', loss, batch_size=val_batch[0].shape[0], prog_bar=True, sync_dist=True)
        self._calculate_metrics(output, target, self.val_metrics)
        return loss

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int]=None) -> Sequence[TimeSeries]:
        if False:
            while True:
                i = 10
        "performs the prediction step\n\n        batch\n            output of Darts' :class:`InferenceDataset` - tuple of ``(past_target, past_covariates,\n            historic_future_covariates, future_covariates, future_past_covariates, input_timeseries)``\n        batch_idx\n            the batch index of the current batch\n        dataloader_idx\n            the dataloader index\n        "
        (input_data_tuple, batch_input_series, batch_pred_starts) = (batch[:-2], batch[-2], batch[-1])
        num_series = input_data_tuple[0].shape[0]
        batch_sample_size = min(max(self.pred_batch_size // num_series, 1), self.pred_num_samples)
        sample_count = 0
        batch_predictions = []
        while sample_count < self.pred_num_samples:
            if sample_count + batch_sample_size > self.pred_num_samples:
                batch_sample_size = self.pred_num_samples - sample_count
            input_data_tuple_samples = self._sample_tiling(input_data_tuple, batch_sample_size)
            batch_prediction = self._get_batch_prediction(self.pred_n, input_data_tuple_samples, self.pred_roll_size)
            out_shape = batch_prediction.shape
            batch_prediction = batch_prediction.reshape((batch_sample_size, num_series) + out_shape[1:])
            batch_predictions.append(batch_prediction)
            sample_count += batch_sample_size
        batch_predictions = torch.cat(batch_predictions, dim=0)
        batch_predictions = batch_predictions.cpu().detach().numpy()
        ts_forecasts = Parallel(n_jobs=self.pred_n_jobs)((delayed(_build_forecast_series)([batch_prediction[batch_idx] for batch_prediction in batch_predictions], input_series, custom_columns=self.likelihood.likelihood_components_names(input_series) if self.predict_likelihood_parameters else None, with_static_covs=False if self.predict_likelihood_parameters else True, with_hierarchy=False if self.predict_likelihood_parameters else True, pred_start=pred_start) for (batch_idx, (input_series, pred_start)) in enumerate(zip(batch_input_series, batch_pred_starts))))
        return ts_forecasts

    def set_predict_parameters(self, n: int, num_samples: int, roll_size: int, batch_size: int, n_jobs: int, predict_likelihood_parameters: bool) -> None:
        if False:
            print('Hello World!')
        'to be set from TorchForecastingModel before calling trainer.predict() and reset at self.on_predict_end()'
        self.pred_n = n
        self.pred_num_samples = num_samples
        self.pred_roll_size = roll_size
        self.pred_batch_size = batch_size
        self.pred_n_jobs = n_jobs
        self.predict_likelihood_parameters = predict_likelihood_parameters

    def _compute_loss(self, output, target):
        if False:
            while True:
                i = 10
        if self.likelihood:
            return self.likelihood.compute_loss(output, target)
        else:
            return self.criterion(output.squeeze(dim=-1), target)

    def _calculate_metrics(self, output, target, metrics):
        if False:
            for i in range(10):
                print('nop')
        if not len(metrics):
            return
        if self.likelihood:
            _metric = metrics(self.likelihood.sample(output), target)
        else:
            _metric = metrics(output.squeeze(dim=-1), target)
        self.log_dict(_metric, on_epoch=True, on_step=False, logger=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if False:
            return 10
        'configures optimizers and learning rate schedulers for model optimization.'

        def _create_from_cls_and_kwargs(cls, kws):
            if False:
                while True:
                    i = 10
            try:
                return cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(ValueError('Error when building the optimizer or learning rate scheduler;please check the provided class and arguments\nclass: {}\narguments (kwargs): {}\nerror:\n{}'.format(cls, kws, e)), logger)
        optimizer_kws = {k: v for (k, v) in self.optimizer_kwargs.items()}
        optimizer_kws['params'] = self.parameters()
        optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)
        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for (k, v) in self.lr_scheduler_kwargs.items()}
            lr_sched_kws['optimizer'] = optimizer
            lr_monitor = lr_sched_kws.pop('monitor', None)
            lr_scheduler = _create_from_cls_and_kwargs(self.lr_scheduler_cls, lr_sched_kws)
            return ([optimizer], {'scheduler': lr_scheduler, 'monitor': lr_monitor if lr_monitor is not None else 'val_loss'})
        else:
            return optimizer

    @abstractmethod
    def _produce_train_output(self, input_batch: Tuple) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        In charge of applying the recurrent logic for non-recurrent models.\n        Should be overwritten by recurrent models.\n        '
        pass

    @staticmethod
    def _sample_tiling(input_data_tuple, batch_sample_size):
        if False:
            i = 10
            return i + 15
        tiled_input_data = []
        for tensor in input_data_tuple:
            if tensor is not None:
                tiled_input_data.append(tensor.tile((batch_sample_size, 1, 1)))
            else:
                tiled_input_data.append(None)
        return tuple(tiled_input_data)

    def _get_mc_dropout_modules(self) -> set:
        if False:
            i = 10
            return i + 15

        def recurse_children(children, acc):
            if False:
                i = 10
                return i + 15
            for module in children:
                if isinstance(module, MonteCarloDropout):
                    acc.add(module)
                acc = recurse_children(module.children(), acc)
            return acc
        return recurse_children(self.children(), set())

    def set_mc_dropout(self, active: bool):
        if False:
            i = 10
            return i + 15
        for module in self._get_mc_dropout_modules():
            module.mc_dropout_enabled = active

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.likelihood is not None or len(self._get_mc_dropout_modules()) > 0

    def _produce_predict_output(self, x: Tuple) -> torch.Tensor:
        if False:
            return 10
        if self.likelihood:
            output = self(x)
            if self.predict_likelihood_parameters:
                return self.likelihood.predict_likelihood_parameters(output)
            else:
                return self.likelihood.sample(output)
        else:
            return self(x).squeeze(dim=-1)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        checkpoint['model_dtype'] = self.dtype
        checkpoint['train_sample_shape'] = self.train_sample_shape
        checkpoint['loss_fn'] = self.criterion
        checkpoint['torch_metrics_train'] = self.train_metrics
        checkpoint['torch_metrics_val'] = self.val_metrics

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        dtype = checkpoint['model_dtype']
        self.to_dtype(dtype)
        if 'loss_fn' in checkpoint.keys() and 'torch_metrics_train' in checkpoint.keys():
            self.criterion = checkpoint['loss_fn']
            self.train_metrics = checkpoint['torch_metrics_train']
            self.val_metrics = checkpoint['torch_metrics_val']
        else:
            logger.warning("This checkpoint was generated with darts <= 0.24.0, if a custom loss was used to train the model, it won't be properly loaded. Similarly, the torch metrics won't be restored from the checkpoint.")

    def to_dtype(self, dtype):
        if False:
            while True:
                i = 10
        'Cast module precision (float32 by default) to another precision.'
        if dtype == torch.float16:
            self.half()
        if dtype == torch.float32:
            self.float()
        elif dtype == torch.float64:
            self.double()
        else:
            raise_if(True, f'Trying to load dtype {dtype}. Loading for this type is not implemented yet. Please report this issue on https://github.com/unit8co/darts', logger)

    @property
    def epochs_trained(self):
        if False:
            i = 10
            return i + 15
        current_epoch = self.current_epoch
        if not pl_160_or_above and (self.current_epoch or self.global_step):
            current_epoch += 1
        return current_epoch

    @property
    def output_chunk_length(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of time steps predicted at once by the model.\n        '
        return self._output_chunk_length

    @staticmethod
    def configure_torch_metrics(torch_metrics: Union[torchmetrics.Metric, torchmetrics.MetricCollection]) -> torchmetrics.MetricCollection:
        if False:
            while True:
                i = 10
        'process the torch_metrics parameter.'
        if torch_metrics is None:
            torch_metrics = torchmetrics.MetricCollection([])
        elif isinstance(torch_metrics, torchmetrics.Metric):
            torch_metrics = torchmetrics.MetricCollection([torch_metrics])
        elif isinstance(torch_metrics, torchmetrics.MetricCollection):
            pass
        else:
            raise_log(AttributeError('`torch_metrics` only accepts type torchmetrics.Metric or torchmetrics.MetricCollection'), logger)
        return torch_metrics

class PLPastCovariatesModule(PLForecastingModule, ABC):

    def _produce_train_output(self, input_batch: Tuple):
        if False:
            while True:
                i = 10
        '\n        Feeds PastCovariatesTorchModel with input and output chunks of a PastCovariatesSequentialDataset for\n        training.\n\n        Parameters:\n        ----------\n        input_batch\n            ``(past_target, past_covariates, static_covariates)``\n        '
        (past_target, past_covariates, static_covariates) = input_batch
        inpt = (torch.cat([past_target, past_covariates], dim=2) if past_covariates is not None else past_target, static_covariates)
        return self(inpt)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Feeds PastCovariatesTorchModel with input and output chunks of a PastCovariatesSequentialDataset to forecast\n        the next ``n`` target values per target variable.\n\n        Parameters:\n        ----------\n        n\n            prediction length\n        input_batch\n            ``(past_target, past_covariates, future_past_covariates, static_covariates)``\n        roll_size\n            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to\n            ``self.output_chunk_length``\n        '
        dim_component = 2
        (past_target, past_covariates, future_past_covariates, static_covariates) = input_batch
        n_targets = past_target.shape[dim_component]
        n_past_covs = past_covariates.shape[dim_component] if past_covariates is not None else 0
        input_past = torch.cat([ds for ds in [past_target, past_covariates] if ds is not None], dim=dim_component)
        out = self._produce_predict_output(x=(input_past, static_covariates))[:, self.first_prediction_index:, :]
        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size
        while prediction_length < n:
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = prediction_length + self.output_chunk_length - n
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]
            input_past = torch.roll(input_past, -roll_size, 1)
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length:, :]
            if self.input_chunk_length >= roll_size:
                (left_past, right_past) = (prediction_length - roll_size, prediction_length)
            else:
                (left_past, right_past) = (prediction_length - self.input_chunk_length, prediction_length)
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets:n_targets + n_past_covs] = future_past_covariates[:, left_past:right_past, :]
            elif n_past_covs:
                input_past[:, :, n_targets:n_targets + n_past_covs] = future_past_covariates[:, left_past:right_past, :]
            out = self._produce_predict_output(x=(input_past, static_covariates))[:, self.first_prediction_index:, :]
            batch_prediction.append(out)
            prediction_length += self.output_chunk_length
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction

class PLFutureCovariatesModule(PLForecastingModule, ABC):

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")

class PLDualCovariatesModule(PLForecastingModule, ABC):

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('TBD: The only DualCovariatesModel is an RNN with a specific implementation.')

class PLMixedCovariatesModule(PLForecastingModule, ABC):

    def _produce_train_output(self, input_batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            return 10
        '\n        Feeds MixedCovariatesTorchModel with input and output chunks of a MixedCovariatesSequentialDataset for\n        training.\n\n        Parameters:\n        ----------\n        input_batch\n            ``(past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates)``.\n        '
        return self(self._process_input_batch(input_batch))

    def _process_input_batch(self, input_batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if False:
            print('Hello World!')
        '\n        Converts output of MixedCovariatesDataset (training dataset) into an input/past- and\n        output/future chunk.\n\n        Parameters\n        ----------\n        input_batch\n            ``(past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates)``.\n\n        Returns\n        -------\n        tuple\n            ``(x_past, x_future, x_static)`` the input/past and output/future chunks.\n        '
        (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates) = input_batch
        dim_variable = 2
        x_past = torch.cat([tensor for tensor in [past_target, past_covariates, historic_future_covariates] if tensor is not None], dim=dim_variable)
        return (x_past, future_covariates, static_covariates)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Feeds MixedCovariatesModel with input and output chunks of a MixedCovariatesSequentialDataset to forecast\n        the next ``n`` target values per target variable.\n\n        Parameters\n        ----------\n        n\n            prediction length\n        input_batch\n            (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates)\n        roll_size\n            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to\n            ``self.output_chunk_length``\n        '
        dim_component = 2
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates, static_covariates) = input_batch
        n_targets = past_target.shape[dim_component]
        n_past_covs = past_covariates.shape[dim_component] if past_covariates is not None else 0
        n_future_covs = future_covariates.shape[dim_component] if future_covariates is not None else 0
        (input_past, input_future, input_static) = self._process_input_batch((past_target, past_covariates, historic_future_covariates, future_covariates[:, :roll_size, :] if future_covariates is not None else None, static_covariates))
        out = self._produce_predict_output(x=(input_past, input_future, input_static))[:, self.first_prediction_index:, :]
        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size
        while prediction_length < n:
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = prediction_length + self.output_chunk_length - n
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]
            input_past = torch.roll(input_past, -roll_size, 1)
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length:, :]
            if self.input_chunk_length >= roll_size:
                (left_past, right_past) = (prediction_length - roll_size, prediction_length)
            else:
                (left_past, right_past) = (prediction_length - self.input_chunk_length, prediction_length)
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets:n_targets + n_past_covs] = future_past_covariates[:, left_past:right_past, :]
            elif n_past_covs:
                input_past[:, :, n_targets:n_targets + n_past_covs] = future_past_covariates[:, left_past:right_past, :]
            if n_future_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets + n_past_covs:] = future_covariates[:, left_past:right_past, :]
            elif n_future_covs:
                input_past[:, :, n_targets + n_past_covs:] = future_covariates[:, left_past:right_past, :]
            (left_future, right_future) = (right_past, right_past + self.output_chunk_length)
            if n_future_covs:
                input_future = future_covariates[:, left_future:right_future, :]
            out = self._produce_predict_output(x=(input_past, input_future, input_static))[:, self.first_prediction_index:, :]
            batch_prediction.append(out)
            prediction_length += self.output_chunk_length
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction

class PLSplitCovariatesModule(PLForecastingModule, ABC):

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")