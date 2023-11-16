"""
TorchForecastingModel

This file contains several abstract classes:

    * TorchForecastingModel is the super-class of all torch (deep learning) darts forecasting models.

    * PastCovariatesTorchModel(TorchForecastingModel) for torch models consuming only past-observed covariates.
    * FutureCovariatesTorchModel(TorchForecastingModel) for torch models consuming only future values of
      future covariates.
    * DualCovariatesTorchModel(TorchForecastingModel) for torch models consuming past and future values of some single
      future covariates.
    * MixedCovariatesTorchModel(TorchForecastingModel) for torch models consuming both past-observed
      as well as past and future values of some future covariates.
    * SplitCovariatesTorchModel(TorchForecastingModel) for torch models consuming past-observed as well as future
      values of some future covariates.

    * TorchParametricProbabilisticForecastingModel(TorchForecastingModel) is the super-class of all probabilistic torch
      forecasting models.
"""
import copy
import datetime
import inspect
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ProgressBar
from torch import Tensor
from torch.utils.data import DataLoader
from darts.dataprocessing.encoders import SequentialEncoder
from darts.logging import get_logger, raise_if, raise_if_not, raise_log, suppress_lightning_warnings
from darts.models.forecasting.forecasting_model import ForecastingModel, GlobalForecastingModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.timeseries import TimeSeries
from darts.utils.data.inference_dataset import DualCovariatesInferenceDataset, FutureCovariatesInferenceDataset, InferenceDataset, MixedCovariatesInferenceDataset, PastCovariatesInferenceDataset, SplitCovariatesInferenceDataset
from darts.utils.data.sequential_dataset import DualCovariatesSequentialDataset, FutureCovariatesSequentialDataset, MixedCovariatesSequentialDataset, PastCovariatesSequentialDataset, SplitCovariatesSequentialDataset
from darts.utils.data.training_dataset import DualCovariatesTrainingDataset, FutureCovariatesTrainingDataset, MixedCovariatesTrainingDataset, PastCovariatesTrainingDataset, SplitCovariatesTrainingDataset, TrainingDataset
from darts.utils.historical_forecasts import _check_optimizable_historical_forecasts_global_models, _process_historical_forecast_input
from darts.utils.historical_forecasts.optimized_historical_forecasts_torch import _optimized_historical_forecasts
from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method
from darts.utils.utils import get_single_series, seq2series, series2seq
tokens = pl.__version__.split('.')
pl_200_or_above = int(tokens[0]) >= 2
if pl_200_or_above:
    from pytorch_lightning.tuner import Tuner
else:
    from pytorch_lightning.tuner.tuning import Tuner
DEFAULT_DARTS_FOLDER = 'darts_logs'
CHECKPOINTS_FOLDER = 'checkpoints'
RUNS_FOLDER = 'runs'
INIT_MODEL_NAME = '_model.pth.tar'
TORCH_NP_DTYPES = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64}
TFM_ATTRS_NO_PICKLE = {'model': None, 'trainer': None}
logger = get_logger(__name__)

def _get_checkpoint_folder(work_dir, model_name):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(work_dir, model_name, CHECKPOINTS_FOLDER)

def _get_logs_folder(work_dir, model_name):
    if False:
        return 10
    return os.path.join(work_dir, model_name)

def _get_runs_folder(work_dir, model_name):
    if False:
        print('Hello World!')
    return os.path.join(work_dir, model_name)

def _get_checkpoint_fname(work_dir, model_name, best=False):
    if False:
        while True:
            i = 10
    checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
    path = os.path.join(checkpoint_dir, 'best-*' if best else 'last-*')
    checklist = glob(path)
    if len(checklist) == 0:
        raise_log(FileNotFoundError('There is no file matching prefix {} in {}'.format('best-*' if best else 'last-*', checkpoint_dir)), logger)
    file_name = max(checklist, key=os.path.getctime)
    return os.path.basename(file_name)

class TorchForecastingModel(GlobalForecastingModel, ABC):

    @random_method
    def __init__(self, batch_size: int=32, n_epochs: int=100, model_name: str=None, work_dir: str=os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER), log_tensorboard: bool=False, nr_epochs_val_period: int=1, force_reset: bool=False, save_checkpoints: bool=False, add_encoders: Optional[dict]=None, random_state: Optional[int]=None, pl_trainer_kwargs: Optional[dict]=None, show_warnings: bool=False):
        if False:
            return 10
        'Pytorch Lightning (PL)-based Forecasting Model.\n\n        This class is meant to be inherited to create a new PL-based forecasting model.\n        It governs the interactions between:\n            - Darts forecasting models (module) :class:`PLTorchForecastingModel`\n            - Darts integrated PL Lightning Trainer :class:`pytorch_lightning.Trainer` or custom PL Trainers\n            - Dataset loaders :class:`TrainingDataset` and :class:`InferenceDataset` or custom Dataset Loaders.\n\n        When subclassing this class, please make sure to set the self.model attribute\n        in the __init__ function and then call super().__init__ while passing the kwargs.\n\n        Parameters\n        ----------\n        batch_size\n            Number of time series (input and output sequences) used in each training pass. Default: ``32``.\n        n_epochs\n            Number of epochs over which to train the model. Default: ``100``.\n        model_name\n            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,\n            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part\n            of the name is formatted with the local date and time, while PID is the processed ID (preventing models\n            spawned at the same time by different processes to share the same model_name). E.g.,\n            ``"2021-06-14_09_53_32_torch_model_run_44607"``.\n        work_dir\n            Path of the working directory, where to save checkpoints and Tensorboard summaries.\n            Default: current working directory.\n        log_tensorboard\n            If set, use Tensorboard to log the different parameters. The logs will be located in:\n            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.\n        nr_epochs_val_period\n            Number of epochs to wait before evaluating the validation loss (if a validation\n            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.\n        force_reset\n            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will\n            be discarded). Default: ``False``.\n        save_checkpoints\n            Whether or not to automatically save the untrained model and checkpoints from training.\n            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where\n            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,\n            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using\n            :func:`save()` and loaded using :func:`load()`. Default: ``False``.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        random_state\n            Control the randomness of the weights initialization. Check this\n            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.\n            Default: ``None``.\n        pl_trainer_kwargs\n            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets\n            that performs the training, validation and prediction processes. These presets include automatic\n            checkpointing, tensorboard logging, setting the torch device and more.\n            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer\n            object. Check the `PL Trainer documentation\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the\n            supported kwargs. Default: ``None``.\n            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",\n            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``\n            dict:\n\n\n            - ``{"accelerator": "cpu"}`` for CPU,\n            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),\n            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.\n\n            For more info, see here:\n            `trainer flags\n            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`_,\n            and `training on multiple gpus\n            <https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus>`_.\n\n            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts\'\n            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.\n            The model will stop training early if the validation loss `val_loss` does not improve beyond\n            specifications. For more information on callbacks, visit:\n            `PyTorch Lightning Callbacks\n            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from pytorch_lightning.callbacks.early_stopping import EarlyStopping\n\n                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over\n                # a period of 5 epochs (`patience`)\n                my_stopper = EarlyStopping(\n                    monitor="val_loss",\n                    patience=5,\n                    min_delta=0.05,\n                    mode=\'min\',\n                )\n\n                pl_trainer_kwargs={"callbacks": [my_stopper]}\n            ..\n\n            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional\n            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.\n        show_warnings\n            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of\n            your forecasting use case. Default: ``False``.\n        '
        super().__init__(add_encoders=add_encoders)
        suppress_lightning_warnings(suppress_all=not show_warnings)
        self.model: Optional[PLForecastingModule] = None
        self._module_path = self.__module__
        self._module_name: Optional[str] = ''
        self.train_sample: Optional[Tuple] = None
        self.output_dim: Optional[int] = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if model_name is None:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            model_name = current_time + '_torch_model_run_' + str(os.getpid())
        self.model_name = model_name
        self.work_dir = work_dir
        self.save_checkpoints = save_checkpoints
        checkpoints_folder = _get_checkpoint_folder(self.work_dir, self.model_name)
        log_folder = _get_logs_folder(self.work_dir, self.model_name)
        checkpoint_exists = os.path.exists(checkpoints_folder) and len(glob(os.path.join(checkpoints_folder, '*'))) > 0
        if checkpoint_exists and save_checkpoints:
            raise_if_not(force_reset, f"Some model data already exists for `model_name` '{self.model_name}'. Either load model to continue training or use `force_reset=True` to initialize anyway to start training from scratch and remove all the model data", logger)
            self.reset_model()
        elif save_checkpoints:
            self._create_save_dirs()
        else:
            pass
        if save_checkpoints:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoints_folder, save_last=True, monitor='val_loss', filename='best-{epoch}-{val_loss:.2f}')
            checkpoint_callback.CHECKPOINT_NAME_LAST = 'last-{epoch}'
        else:
            checkpoint_callback = None
        model_logger = pl_loggers.TensorBoardLogger(save_dir=log_folder, name='', version='logs') if log_tensorboard else False
        self.trainer_params: Dict[str, Any] = {'logger': model_logger, 'max_epochs': n_epochs, 'check_val_every_n_epoch': nr_epochs_val_period, 'enable_checkpointing': save_checkpoints, 'callbacks': [cb for cb in [checkpoint_callback] if cb is not None]}
        if pl_trainer_kwargs is not None:
            pl_trainer_kwargs_copy = {key: val for (key, val) in pl_trainer_kwargs.items()}
            self.n_epochs = pl_trainer_kwargs_copy.get('max_epochs', self.n_epochs)
            self.trainer_params['callbacks'] += pl_trainer_kwargs_copy.pop('callbacks', [])
            self.trainer_params = dict(self.trainer_params, **pl_trainer_kwargs_copy)
        self.trainer: Optional[pl.Trainer] = None
        self.load_ckpt_path: Optional[str] = None
        self.pl_module_params: Optional[dict] = None

    @classmethod
    def _validate_model_params(cls, **kwargs):
        if False:
            while True:
                i = 10
        'validate that parameters used at model creation are part of :class:`TorchForecastingModel`,\n        :class:`PLForecastingModule` or cls __init__ methods.\n        '
        valid_kwargs = set(inspect.signature(TorchForecastingModel.__init__).parameters.keys()) | set(inspect.signature(PLForecastingModule.__init__).parameters.keys()) | set(inspect.signature(cls.__init__).parameters.keys())
        invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in valid_kwargs]
        raise_if(len(invalid_kwargs) > 0, f'Invalid model creation parameters. Model `{cls.__name__}` has no args/kwargs `{invalid_kwargs}`', logger=logger)

    @classmethod
    def _extract_torch_model_params(cls, **kwargs):
        if False:
            return 10
        'extract params from model creation to set up TorchForecastingModels'
        cls._validate_model_params(**kwargs)
        get_params = list(inspect.signature(TorchForecastingModel.__init__).parameters.keys())
        get_params.remove('self')
        return {kwarg: kwargs.get(kwarg) for kwarg in get_params if kwarg in kwargs}

    @staticmethod
    def _extract_pl_module_params(**kwargs):
        if False:
            print('Hello World!')
        'Extract params from model creation to set up PLForecastingModule (the actual torch.nn.Module)'
        get_params = list(inspect.signature(PLForecastingModule.__init__).parameters.keys())
        get_params.remove('self')
        return {kwarg: kwargs.get(kwarg) for kwarg in get_params if kwarg in kwargs}

    def _create_save_dirs(self):
        if False:
            print('Hello World!')
        'Create work dir and model dir'
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        if not os.path.exists(_get_runs_folder(self.work_dir, self.model_name)):
            os.mkdir(_get_runs_folder(self.work_dir, self.model_name))

    def _remove_save_dirs(self):
        if False:
            print('Hello World!')
        shutil.rmtree(_get_runs_folder(self.work_dir, self.model_name), ignore_errors=True)

    def reset_model(self):
        if False:
            i = 10
            return i + 15
        'Resets the model object and removes all stored data - model, checkpoints, loggers and training history.'
        self._remove_save_dirs()
        self._create_save_dirs()
        self.model = None
        self.train_sample = None

    def _init_model(self, trainer: Optional[pl.Trainer]=None) -> PLForecastingModule:
        if False:
            while True:
                i = 10
        'Initializes model and trainer based on examples of input/output tensors (to get the sizes right):'
        raise_if(self.pl_module_params is None, '`pl_module_params` must be extracted in __init__ method of `TorchForecastingModel` subclass after calling `super.__init__(...)`. Do this with `self._extract_pl_module_params(**self.model_params).`')
        self.pl_module_params['train_sample_shape'] = [variate.shape if variate is not None else None for variate in self.train_sample]
        model = self._create_model(self.train_sample)
        self._module_name = model.__class__.__name__
        precision = None
        dtype = self.train_sample[0].dtype
        if np.issubdtype(dtype, np.float32):
            logger.info('Time series values are 32-bits; casting model to float32.')
            precision = '32' if not pl_200_or_above else '32-true'
        elif np.issubdtype(dtype, np.float64):
            logger.info('Time series values are 64-bits; casting model to float64.')
            precision = '64' if not pl_200_or_above else '64-true'
        else:
            raise_log(ValueError(f'Invalid time series data type `{dtype}`. Cast your data to `np.float32` or `np.float64`, e.g. with `TimeSeries.astype(np.float32)`.'), logger)
        precision_int = int(re.findall('\\d+', str(precision))[0])
        precision_user = self.trainer_params.get('precision', None) if trainer is None else trainer.precision
        if precision_user is not None:
            valid_precisions = ['64', '32'] if not pl_200_or_above else ['64-true', '32-true']
            if str(precision_user) not in valid_precisions:
                raise_log(ValueError(f'Invalid user-defined trainer_kwarg `precision={precision_user}`. Use one of ({valid_precisions})'), logger)
            precision_user_int = int(re.findall('\\d+', str(precision_user))[0])
        else:
            precision_user_int = None
        raise_if(precision_user is not None and precision_user_int != precision_int, f"User-defined trainer_kwarg `precision='{precision_user}'` does not match dtype: `{dtype}` of the underlying TimeSeries. Set `precision` to `{precision}` or cast your data to `{precision_user}` with `TimeSeries.astype(np.float{precision_user_int})`.", logger)
        self.trainer_params['precision'] = precision
        if self.save_checkpoints:
            self.save(os.path.join(_get_runs_folder(self.work_dir, self.model_name), INIT_MODEL_NAME))
        return model

    def _setup_trainer(self, trainer: Optional[pl.Trainer], model: PLForecastingModule, verbose: Optional[bool]=None, epochs: int=0) -> pl.Trainer:
        if False:
            print('Hello World!')
        'Sets up a PyTorch-Lightning trainer (if not already provided) for training or prediction.'
        if trainer is not None:
            return trainer
        trainer_params = {key: val for (key, val) in self.trainer_params.items()}
        has_progress_bar = any([isinstance(cb, ProgressBar) for cb in trainer_params.get('callbacks', [])])
        if verbose is not None and (not has_progress_bar):
            trainer_params['enable_model_summary'] = verbose if model.epochs_trained == 0 else False
            trainer_params['enable_progress_bar'] = verbose
        return self._init_trainer(trainer_params=trainer_params, max_epochs=epochs)

    @staticmethod
    def _init_trainer(trainer_params: dict, max_epochs: Optional[int]=None) -> pl.Trainer:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a PyTorch-Lightning trainer for training or prediction from `trainer_params`.'
        trainer_params_copy = {key: val for (key, val) in trainer_params.items()}
        if max_epochs is not None:
            trainer_params_copy['max_epochs'] = max_epochs
        callbacks = trainer_params_copy.pop('callbacks', None)
        return pl.Trainer(callbacks=[cb for cb in callbacks] if callbacks is not None else callbacks, **trainer_params_copy)

    @abstractmethod
    def _create_model(self, train_sample: Tuple[Tensor]) -> PLForecastingModule:
        if False:
            while True:
                i = 10
        '\n        This method has to be implemented by all children. It is in charge of instantiating the actual torch model,\n        based on examples input/output tensors (i.e. implement a model with the right input/output sizes).\n        '
        pass

    @abstractmethod
    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> TrainingDataset:
        if False:
            for i in range(10):
                print('nop')
        '\n        Each model must specify the default training dataset to use.\n        '
        pass

    @abstractmethod
    def _build_inference_dataset(self, target: Sequence[TimeSeries], n: int, past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], stride: int=0, bounds: Optional[np.ndarray]=None) -> InferenceDataset:
        if False:
            i = 10
            return i + 15
        '\n        Each model must specify the default training dataset to use.\n        '
        pass

    @abstractmethod
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that the provided train dataset is of the correct type\n        '
        pass

    @abstractmethod
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        if False:
            print('Hello World!')
        '\n        Verify that the provided inference dataset is of the correct type\n        '
        pass

    @abstractmethod
    def _verify_predict_sample(self, predict_sample: Tuple):
        if False:
            i = 10
            return i + 15
        '\n        verify that the (first) sample contained in the inference dataset matches the model type and the\n        data the model has been trained on.\n        '
        pass

    @abstractmethod
    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that any non-None covariates comply with the model type.\n        '
        pass

    @random_method
    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, trainer: Optional[pl.Trainer]=None, verbose: Optional[bool]=None, epochs: int=0, max_samples_per_ts: Optional[int]=None, num_loader_workers: int=0) -> 'TorchForecastingModel':
        if False:
            print('Hello World!')
        "Fit/train the model on one or multiple series.\n\n        This method wraps around :func:`fit_from_dataset()`, constructing a default training\n        dataset for this model. If you need more control on how the series are sliced for training, consider\n        calling :func:`fit_from_dataset()` with a custom :class:`darts.utils.data.TrainingDataset`.\n\n        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and\n        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter\n        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link\n        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .\n\n        This function can be called several times to do some extra training. If ``epochs`` is specified, the model\n        will be trained for some (extra) ``epochs`` epochs.\n\n        Below, all possible parameters are documented, but not all models support all parameters. For instance,\n        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.\n        Darts will complain if you try fitting a model with the wrong covariates argument.\n\n        When handling covariates, Darts will try to use the time axes of the target and the covariates\n        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes\n        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.\n\n        Parameters\n        ----------\n        series\n            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)\n        past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates\n        future_covariates\n            Optionally, a series or sequence of series specifying future-known covariates\n        val_series\n            Optionally, one or a sequence of validation target series, which will be used to compute the validation\n            loss throughout training and keep track of the best performing models.\n        val_past_covariates\n            Optionally, the past covariates corresponding to the validation series (must match ``covariates``)\n        val_future_covariates\n            Optionally, the future covariates corresponding to the validation series (must match ``covariates``)\n        trainer\n            Optionally, a custom PyTorch-Lightning Trainer object to perform training. Using a custom ``trainer`` will\n            override Darts' default trainer.\n        verbose\n            Optionally, whether to print the progress. Ignored if there is a `ProgressBar` callback in\n            `pl_trainer_kwargs`.\n        epochs\n            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``\n            was provided to the model constructor.\n        max_samples_per_ts\n            Optionally, a maximum number of samples to use per time series. Models are trained in a supervised fashion\n            by constructing slices of (input, output) examples. On long time series, this can result in unnecessarily\n            large number of training samples. This parameter upper-bounds the number of training samples per time\n            series (taking only the most recent samples in each series). Leaving to None does not apply any\n            upper bound.\n        num_loader_workers\n            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,\n            both for the training and validation loaders (if any).\n            A larger number of workers can sometimes increase performance, but can also incur extra overheads\n            and increase memory usage, as more batches are loaded in parallel.\n\n        Returns\n        -------\n        self\n            Fitted model.\n        "
        ((series, past_covariates, future_covariates), params) = self._setup_for_fit_from_dataset(series=series, past_covariates=past_covariates, future_covariates=future_covariates, val_series=val_series, val_past_covariates=val_past_covariates, val_future_covariates=val_future_covariates, trainer=trainer, verbose=verbose, epochs=epochs, max_samples_per_ts=max_samples_per_ts, num_loader_workers=num_loader_workers)
        super().fit(series=seq2series(series), past_covariates=seq2series(past_covariates), future_covariates=seq2series(future_covariates))
        return self.fit_from_dataset(*params)

    def _setup_for_fit_from_dataset(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, trainer: Optional[pl.Trainer]=None, verbose: Optional[bool]=None, epochs: int=0, max_samples_per_ts: Optional[int]=None, num_loader_workers: int=0) -> Tuple[Tuple[Sequence[TimeSeries], Optional[Sequence[TimeSeries]], Optional[Sequence[TimeSeries]]], Tuple[TrainingDataset, Optional[TrainingDataset], Optional[pl.Trainer], Optional[bool], int, int]]:
        if False:
            while True:
                i = 10
        'This method acts on `TimeSeries` inputs. It performs sanity checks, and sets up / returns the datasets and\n        additional inputs required for training the model with `fit_from_dataset()`.\n        '
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        val_series = series2seq(val_series)
        val_past_covariates = series2seq(val_past_covariates)
        val_future_covariates = series2seq(val_future_covariates)
        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            (past_covariates, future_covariates) = self.generate_fit_encodings(series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True
        if get_single_series(series).static_covariates is not None and self.supports_static_covariates and self.considers_static_covariates:
            self._uses_static_covariates = True
        self._verify_past_future_covariates(past_covariates=past_covariates, future_covariates=future_covariates)
        self._verify_static_covariates(series[0].static_covariates)
        if val_series is not None:
            if self.encoders.encoding_available:
                (val_past_covariates, val_future_covariates) = self.generate_fit_encodings(series=val_series, past_covariates=val_past_covariates, future_covariates=val_future_covariates)
            self._verify_past_future_covariates(past_covariates=val_past_covariates, future_covariates=val_future_covariates)
            self._verify_static_covariates(val_series[0].static_covariates)
            match = series[0].width == val_series[0].width and (past_covariates[0].width if past_covariates is not None else None) == (val_past_covariates[0].width if val_past_covariates is not None else None) and ((future_covariates[0].width if future_covariates is not None else None) == (val_future_covariates[0].width if val_future_covariates is not None else None))
            raise_if_not(match, 'The dimensions of the series in the training set and the validation set do not match.')
        train_dataset = self._build_train_dataset(target=series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts)
        if val_series is not None:
            val_dataset = self._build_train_dataset(target=val_series, past_covariates=val_past_covariates, future_covariates=val_future_covariates, max_samples_per_ts=max_samples_per_ts)
        else:
            val_dataset = None
        length_ok = True
        try:
            len(train_dataset)
        except ValueError:
            length_ok = False
        raise_if(not length_ok or len(train_dataset) == 0, 'The train dataset does not contain even one training sample. ' + 'This is likely due to the provided training series being too short. ' + 'This model expect series of length at least {}.'.format(self.min_train_series_length))
        logger.info(f'Train dataset contains {len(train_dataset)} samples.')
        series_input = (series, past_covariates, future_covariates)
        fit_from_ds_params = (train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers)
        return (series_input, fit_from_ds_params)

    @random_method
    def fit_from_dataset(self, train_dataset: TrainingDataset, val_dataset: Optional[TrainingDataset]=None, trainer: Optional[pl.Trainer]=None, verbose: Optional[bool]=None, epochs: int=0, num_loader_workers: int=0) -> 'TorchForecastingModel':
        if False:
            return 10
        "\n        Train the model with a specific :class:`darts.utils.data.TrainingDataset` instance.\n        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced\n        for training. If you are not sure which training dataset to use, consider calling :func:`fit()` instead,\n        which will create a default training dataset appropriate for this model.\n\n        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and\n        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter\n        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link\n        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_.\n\n        This function can be called several times to do some extra training. If ``epochs`` is specified, the model\n        will be trained for some (extra) ``epochs`` epochs.\n\n        Parameters\n        ----------\n        train_dataset\n            A training dataset with a type matching this model (e.g. :class:`PastCovariatesTrainingDataset` for\n            :class:`PastCovariatesTorchModel`).\n        val_dataset\n            A training dataset with a type matching this model (e.g. :class:`PastCovariatesTrainingDataset` for\n            :class:`PastCovariatesTorchModel`s), representing the validation set (to track the validation loss).\n        trainer\n            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom `trainer` will\n            override Darts' default trainer.\n        verbose\n            Optionally, whether to print the progress. Ignored if there is a `ProgressBar` callback in\n            `pl_trainer_kwargs`.\n        epochs\n            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``\n            was provided to the model constructor.\n        num_loader_workers\n            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,\n            both for the training and validation loaders (if any).\n            A larger number of workers can sometimes increase performance, but can also incur extra overheads\n            and increase memory usage, as more batches are loaded in parallel.\n\n        Returns\n        -------\n        self\n            Fitted model.\n        "
        self._train(*self._setup_for_train(train_dataset=train_dataset, val_dataset=val_dataset, trainer=trainer, verbose=verbose, epochs=epochs, num_loader_workers=num_loader_workers))
        return self

    def _setup_for_train(self, train_dataset: TrainingDataset, val_dataset: Optional[TrainingDataset]=None, trainer: Optional[pl.Trainer]=None, verbose: Optional[bool]=None, epochs: int=0, num_loader_workers: int=0) -> Tuple[pl.Trainer, PLForecastingModule, DataLoader, Optional[DataLoader]]:
        if False:
            i = 10
            return i + 15
        'This method acts on `TrainingDataset` inputs. It performs sanity checks, and sets up / returns the trainer,\n        model, and dataset loaders required for training the model with `_train()`.\n        '
        self._verify_train_dataset_type(train_dataset)
        (train_length_ok, val_length_ok) = (True, True)
        try:
            len(train_dataset)
        except ValueError:
            train_length_ok = False
        if val_dataset is not None:
            try:
                len(val_dataset)
            except ValueError:
                val_length_ok = False
        raise_if(not train_length_ok or len(train_dataset) == 0, 'The provided training time series dataset is too short for obtaining even one training point.', logger)
        raise_if(val_dataset is not None and (not val_length_ok or len(val_dataset) == 0), 'The provided validation time series dataset is too short for obtaining even one training point.', logger)
        train_sample = train_dataset[0]
        if self.model is None:
            (self.train_sample, self.output_dim) = (train_sample, train_sample[-1].shape[1])
            model = self._init_model(trainer)
        else:
            model = self.model
            raise_if_not(len(train_sample) == len(self.train_sample), 'The size of the training set samples (tuples) does not match what the model has been previously trained on. Trained on tuples of length {}, received tuples of length {}.'.format(len(self.train_sample), len(train_sample)))
            same_dims = tuple((s.shape[1] if s is not None else None for s in train_sample)) == tuple((s.shape[1] if s is not None else None for s in self.train_sample))
            raise_if_not(same_dims, 'The dimensionality of the series in the training set do not match the dimensionality of the series the model has previously been trained on. Model input/output dimensions = {}, provided input/output dimensions = {}'.format(tuple((s.shape[1] if s is not None else None for s in self.train_sample)), tuple((s.shape[1] if s is not None else None for s in train_sample))))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_loader_workers, pin_memory=True, drop_last=False, collate_fn=self._batch_collate_fn)
        val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_loader_workers, pin_memory=True, drop_last=False, collate_fn=self._batch_collate_fn)
        train_num_epochs = epochs if epochs > 0 else self.n_epochs
        trainer = self._setup_trainer(trainer, model, verbose, train_num_epochs)
        if model.epochs_trained > 0 and (not self.load_ckpt_path):
            logger.warning(f'Attempting to retrain/fine-tune the model without resuming from a checkpoint. This is currently discouraged. Consider model `{self.__class__.__name__}.load_weights()` to load the weights for fine-tuning.')
        return (trainer, model, train_loader, val_loader)

    def _train(self, trainer: pl.Trainer, model: PLForecastingModule, train_loader: DataLoader, val_loader: Optional[DataLoader]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Performs the actual training\n\n        Parameters\n        ----------\n        train_loader\n            the training data loader feeding the training data and targets\n        val_loader\n            optionally, a validation set loader\n        '
        self._fit_called = True
        ckpt_path = self.load_ckpt_path
        self.load_ckpt_path = None
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        self.model = model
        self.trainer = trainer

    @random_method
    def lr_find(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, trainer: Optional[pl.Trainer]=None, verbose: Optional[bool]=None, epochs: int=0, max_samples_per_ts: Optional[int]=None, num_loader_workers: int=0, min_lr: float=1e-08, max_lr: float=1, num_training: int=100, mode: str='exponential', early_stop_threshold: float=4.0):
        if False:
            i = 10
            return i + 15
        '\n        A wrapper around PyTorch Lightning\'s `Tuner.lr_find()`. Performs a range test of good initial learning rates,\n        to reduce the amount of guesswork in picking a good starting learning rate. For more information on PyTorch\n        Lightning\'s Tuner check out\n        `this link <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.tuner.tuning.Tuner.html>`_.\n        It is recommended to increase the number of `epochs` if the tuner did not give satisfactory results.\n        Consider creating a new model object with the suggested learning rate for example using model creation\n        parameters `optimizer_cls`, `optimizer_kwargs`, `lr_scheduler_cls`, and `lr_scheduler_kwargs`.\n\n        Example using a :class:`RNNModel`:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                import torch\n                from darts.datasets import AirPassengersDataset\n                from darts.models import NBEATSModel\n\n                series = AirPassengersDataset().load()\n                train, val = series[:-18], series[-18:]\n                model = NBEATSModel(input_chunk_length=12, output_chunk_length=6, random_state=42)\n                # run the learning rate tuner\n                results = model.lr_find(series=train, val_series=val)\n                # plot the results\n                results.plot(suggest=True, show=True)\n                # create a new model with the suggested learning rate\n                model = NBEATSModel(\n                    input_chunk_length=12,\n                    output_chunk_length=6,\n                    random_state=42,\n                    optimizer_cls=torch.optim.Adam,\n                    optimizer_kwargs={"lr": results.suggestion()}\n                )\n            ..\n\n        Parameters\n        ----------\n        series\n            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)\n        past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates\n        future_covariates\n            Optionally, a series or sequence of series specifying future-known covariates\n        val_series\n            Optionally, one or a sequence of validation target series, which will be used to compute the validation\n            loss throughout training and keep track of the best performing models.\n        val_past_covariates\n            Optionally, the past covariates corresponding to the validation series (must match ``covariates``)\n        val_future_covariates\n            Optionally, the future covariates corresponding to the validation series (must match ``covariates``)\n        trainer\n            Optionally, a custom PyTorch-Lightning Trainer object to perform training. Using a custom ``trainer`` will\n            override Darts\' default trainer.\n        verbose\n            Optionally, whether to print the progress. Ignored if there is a `ProgressBar` callback in\n            `pl_trainer_kwargs`.\n        epochs\n            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``\n            was provided to the model constructor.\n        max_samples_per_ts\n            Optionally, a maximum number of samples to use per time series. Models are trained in a supervised fashion\n            by constructing slices of (input, output) examples. On long time series, this can result in unnecessarily\n            large number of training samples. This parameter upper-bounds the number of training samples per time\n            series (taking only the most recent samples in each series). Leaving to None does not apply any\n            upper bound.\n        num_loader_workers\n            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,\n            both for the training and validation loaders (if any).\n            A larger number of workers can sometimes increase performance, but can also incur extra overheads\n            and increase memory usage, as more batches are loaded in parallel.\n        min_lr\n            minimum learning rate to investigate\n        max_lr\n            maximum learning rate to investigate\n        num_training\n            number of learning rates to test\n        mode\n            Search strategy to update learning rate after each batch:\n            \'exponential\': Increases the learning rate exponentially.\n            \'linear\': Increases the learning rate linearly.\n        early_stop_threshold\n            Threshold for stopping the search. If the loss at any point is larger\n            than early_stop_threshold*best_loss then the search is stopped.\n            To disable, set to `None`\n\n        Returns\n        -------\n        lr_finder\n            `_LRFinder` object of Lightning containing the results of the LR sweep.\n        '
        (_, params) = self._setup_for_fit_from_dataset(series=series, past_covariates=past_covariates, future_covariates=future_covariates, val_series=val_series, val_past_covariates=val_past_covariates, val_future_covariates=val_future_covariates, trainer=trainer, verbose=verbose, epochs=epochs, max_samples_per_ts=max_samples_per_ts, num_loader_workers=num_loader_workers)
        (trainer, model, train_loader, val_loader) = self._setup_for_train(*params)
        return Tuner(trainer).lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, method='fit', min_lr=min_lr, max_lr=max_lr, num_training=num_training, mode=mode, early_stop_threshold=early_stop_threshold, update_attr=False)

    @random_method
    def predict(self, n: int, series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, trainer: Optional[pl.Trainer]=None, batch_size: Optional[int]=None, verbose: Optional[bool]=None, n_jobs: int=1, roll_size: Optional[int]=None, num_samples: int=1, num_loader_workers: int=0, mc_dropout: bool=False, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            return 10
        'Predict the ``n`` time step following the end of the training series, or of the specified ``series``.\n\n        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and\n        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter\n        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link\n        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .\n\n        Below, all possible parameters are documented, but not all models support all parameters. For instance,\n        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.\n        Darts will complain if you try calling :func:`predict()` on a model with the wrong covariates argument.\n\n        Darts will also complain if the provided covariates do not have a sufficient time span.\n        In general, not all models require the same covariates\' time spans:\n\n        * | Models relying on past covariates require the last ``input_chunk_length`` of the ``past_covariates``\n          | points to be known at prediction time. For horizon values ``n > output_chunk_length``, these models\n          | require at least the next ``n - output_chunk_length`` future values to be known as well.\n        * | Models relying on future covariates require the next ``n`` values to be known.\n          | In addition (for :class:`DualCovariatesTorchModel` and :class:`MixedCovariatesTorchModel`), they also\n          | require the "historic" values of these future covariates (over the past ``input_chunk_length``).\n\n        When handling covariates, Darts will try to use the time axes of the target and the covariates\n        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes\n        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.\n\n        Parameters\n        ----------\n        n\n            The number of time steps after the end of the training time series for which to produce predictions\n        series\n            Optionally, a series or sequence of series, representing the history of the target series whose\n            future is to be predicted. If specified, the method returns the forecasts of these\n            series. Otherwise, the method returns the forecast of the (single) training series.\n        past_covariates\n            Optionally, the past-observed covariates series needed as inputs for the model.\n            They must match the covariates used for training in terms of dimension.\n        future_covariates\n            Optionally, the future-known covariates series needed as inputs for the model.\n            They must match the covariates used for training in terms of dimension.\n        trainer\n            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom ``trainer``\n            will override Darts\' default trainer.\n        batch_size\n            Size of batches during prediction. Defaults to the models\' training ``batch_size`` value.\n        verbose\n            Optionally, whether to print the progress. Ignored if there is a `ProgressBar` callback in\n            `pl_trainer_kwargs`.\n        n_jobs\n            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.\n        roll_size\n            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many\n            outputs of the model are fed back into it at every iteration of feeding the predicted target\n            (and optionally future covariates) back into the model. If this parameter is not provided,\n            it will be set ``output_chunk_length`` by default.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1\n            for deterministic models.\n        num_loader_workers\n            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,\n            for the inference/prediction dataset loaders (if any).\n            A larger number of workers can sometimes increase performance, but can also incur extra overheads\n            and increase memory usage, as more batches are loaded in parallel.\n        mc_dropout\n            Optionally, enable monte carlo dropout for predictions using neural network based models.\n            This allows bayesian approximation by specifying an implicit prior over learned models.\n        predict_likelihood_parameters\n            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only\n            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.\n            Default: ``False``\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries]]\n            One or several time series containing the forecasts of ``series``, or the forecast of the training series\n            if ``series`` is not specified and the model has been trained on a single series.\n        '
        if series is None:
            if self.training_series is None:
                raise_log(ValueError('Input `series` must be provided. This is the result either from fitting on multiple series, or from not having fit the model yet.'), logger)
            series = self.training_series
        called_with_single_series = True if isinstance(series, TimeSeries) else False
        series = series2seq(series)
        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = [self.past_covariate_series] * len(series)
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = [self.future_covariate_series] * len(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        self._verify_static_covariates(series[0].static_covariates)
        if self.encoders is not None and self.encoders.encoding_available:
            (past_covariates, future_covariates) = self.generate_predict_encodings(n=n, series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        super().predict(n, series, past_covariates, future_covariates, num_samples=num_samples, predict_likelihood_parameters=predict_likelihood_parameters)
        dataset = self._build_inference_dataset(target=series, n=n, past_covariates=past_covariates, future_covariates=future_covariates, stride=0, bounds=None)
        predictions = self.predict_from_dataset(n, dataset, trainer=trainer, verbose=verbose, batch_size=batch_size, n_jobs=n_jobs, roll_size=roll_size, num_samples=num_samples, num_loader_workers=num_loader_workers, mc_dropout=mc_dropout, predict_likelihood_parameters=predict_likelihood_parameters)
        return predictions[0] if called_with_single_series else predictions

    @random_method
    def predict_from_dataset(self, n: int, input_series_dataset: InferenceDataset, trainer: Optional[pl.Trainer]=None, batch_size: Optional[int]=None, verbose: Optional[bool]=None, n_jobs: int=1, roll_size: Optional[int]=None, num_samples: int=1, num_loader_workers: int=0, mc_dropout: bool=False, predict_likelihood_parameters: bool=False) -> Sequence[TimeSeries]:
        if False:
            print('Hello World!')
        "\n        This method allows for predicting with a specific :class:`darts.utils.data.InferenceDataset` instance.\n        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced\n        for inference. In most cases, you'll rather want to call :func:`predict()` instead, which will create an\n        appropriate :class:`InferenceDataset` for you.\n\n        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and\n        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter\n        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link\n        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .\n\n        Parameters\n        ----------\n        n\n            The number of time steps after the end of the training time series for which to produce predictions\n        input_series_dataset\n            Optionally, a series or sequence of series, representing the history of the target series' whose\n            future is to be predicted. If specified, the method returns the forecasts of these\n            series. Otherwise, the method returns the forecast of the (single) training series.\n        trainer\n            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction.  Using a custom ``trainer``\n            will override Darts' default trainer.\n        batch_size\n            Size of batches during prediction. Defaults to the models ``batch_size`` value.\n        verbose\n            Optionally, whether to print the progress. Ignored if there is a `ProgressBar` callback in\n            `pl_trainer_kwargs`.\n        n_jobs\n            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.\n        roll_size\n            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many\n            outputs of the model are fed back into it at every iteration of feeding the predicted target\n            (and optionally future covariates) back into the model. If this parameter is not provided,\n            it will be set ``output_chunk_length`` by default.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1\n            for deterministic models.\n        num_loader_workers\n            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,\n            for the inference/prediction dataset loaders (if any).\n            A larger number of workers can sometimes increase performance, but can also incur extra overheads\n            and increase memory usage, as more batches are loaded in parallel.\n        mc_dropout\n            Optionally, enable monte carlo dropout for predictions using neural network based models.\n            This allows bayesian approximation by specifying an implicit prior over learned models.\n        predict_likelihood_parameters\n            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only\n            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.\n            Default: ``False``\n\n        Returns\n        -------\n        Sequence[TimeSeries]\n            Returns one or more forecasts for time series.\n        "
        ForecastingModel.predict(self, n, num_samples)
        self._verify_inference_dataset_type(input_series_dataset)
        self._verify_predict_sample(input_series_dataset[0])
        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(0 < roll_size <= self.output_chunk_length, '`roll_size` must be an integer between 1 and `self.output_chunk_length`.')
        raise_if(predict_likelihood_parameters and n > self.output_chunk_length, '`n` must be smaller than or equal to `output_chunk_length` when `predict_likelihood_parameters=True`.', logger)
        raise_if_not(num_samples > 0, '`num_samples` must be a positive integer.')
        batch_size = batch_size or self.batch_size
        self.model.set_predict_parameters(n=n, num_samples=num_samples, roll_size=roll_size, batch_size=batch_size, n_jobs=n_jobs, predict_likelihood_parameters=predict_likelihood_parameters)
        pred_loader = DataLoader(input_series_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers, pin_memory=True, drop_last=False, collate_fn=self._batch_collate_fn)
        self.model.set_mc_dropout(mc_dropout)
        self.trainer = self._setup_trainer(trainer=trainer, model=self.model, verbose=verbose, epochs=self.n_epochs)
        predictions = self.trainer.predict(self.model, pred_loader)
        return [ts for batch in predictions for ts in batch]

    @property
    def first_prediction_index(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns the index of the first predicted within the output of self.model.\n        '
        return 0

    @property
    def min_train_series_length(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Class property defining the minimum required length for the training series;\n        overriding the default value of 3 of ForecastingModel\n        '
        return self.input_chunk_length + self.output_chunk_length

    @staticmethod
    def _batch_collate_fn(batch: List[Tuple]) -> Tuple:
        if False:
            return 10
        '\n        Returns a batch Tuple from a list of samples\n        '
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                aggregated.append(torch.from_numpy(np.stack([sample[i] for sample in batch], axis=0)))
            elif elem is None:
                aggregated.append(None)
            else:
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)

    def save(self, path: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Saves the model under a given path.\n\n        Creates two files under ``path`` (model object) and ``path``.ckpt (checkpoint).\n\n        Example for saving and loading a :class:`RNNModel`:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from darts.models import RNNModel\n\n                model = RNNModel(input_chunk_length=4)\n\n                model.save("my_model.pt")\n                model_loaded = RNNModel.load("my_model.pt")\n            ..\n\n        Parameters\n        ----------\n        path\n            Path under which to save the model at its current state. Please avoid path starting with "last-" or\n            "best-" to avoid collision with Pytorch-Ligthning checkpoints. If no path is specified, the model\n            is automatically saved under ``"{ModelClass}_{YYYY-mm-dd_HH_MM_SS}.pt"``.\n            E.g., ``"RNNModel_2020-01-01_12_00_00.pt"``.\n        '
        if path is None:
            path = self._default_save_path() + '.pt'
        with open(path, 'wb') as f_out:
            torch.save(self, f_out)
        path_ptl_ckpt = path + '.ckpt'
        if self.trainer is not None:
            self.trainer.save_checkpoint(path_ptl_ckpt)
        elif self.load_ckpt_path:
            if os.path.exists(self.load_ckpt_path):
                shutil.copy(self.load_ckpt_path, path_ptl_ckpt)
            else:
                logger.warning(f'Model was not trained since the last loading and attempt to retrieve PyTorch Lightning checkpoint {self.load_ckpt_path} was unsuccessful: model was saved without its weights.')

    @staticmethod
    def load(path: str, **kwargs) -> 'TorchForecastingModel':
        if False:
            for i in range(10):
                print('nop')
        '\n        Loads a model from a given file path.\n\n        Example for loading a general save from :class:`RNNModel`:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from darts.models import RNNModel\n\n                model_loaded = RNNModel.load(path)\n            ..\n\n        Example for loading an :class:`RNNModel` to CPU that was saved on GPU:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from darts.models import RNNModel\n\n                model_loaded = RNNModel.load(path, map_location="cpu")\n                model_loaded.to_cpu()\n            ..\n\n        Parameters\n        ----------\n        path\n            Path from which to load the model. If no path was specified when saving the model, the automatically\n            generated path ending with ".pt" has to be provided.\n        **kwargs\n            Additional kwargs for PyTorch Lightning\'s :func:`LightningModule.load_from_checkpoint()` method,\n            such as ``map_location`` to load the model onto a different device than the one from which it was saved.\n            For more information, read the `official documentation <https://pytorch-lightning.readthedocs.io/en/stable/\n            common/lightning_module.html#load-from-checkpoint>`_.\n        '
        with open(path, 'rb') as fin:
            model: TorchForecastingModel = torch.load(fin, map_location=kwargs.get('map_location', None))
        path_ptl_ckpt = path + '.ckpt'
        if os.path.exists(path_ptl_ckpt):
            model.model = model._load_from_checkpoint(path_ptl_ckpt, **kwargs)
        else:
            model._fit_called = False
            logger.warning(f"Model was loaded without weights since no PyTorch LightningModule checkpoint ('.ckpt') could be found at {path_ptl_ckpt}. Please call `fit()` before calling `predict()`.")
        return model

    @staticmethod
    def load_from_checkpoint(model_name: str, work_dir: str=None, file_name: str=None, best: bool=True, **kwargs) -> 'TorchForecastingModel':
        if False:
            print('Hello World!')
        '\n        Load the model from automatically saved checkpoints under \'{work_dir}/darts_logs/{model_name}/checkpoints/\'.\n        This method is used for models that were created with ``save_checkpoints=True``.\n\n        If you manually saved your model, consider using :meth:`load() <TorchForecastingModel.load()>`.\n\n        Example for loading a :class:`RNNModel` from checkpoint (``model_name`` is the ``model_name`` used at model\n        creation):\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from darts.models import RNNModel\n\n                model_loaded = RNNModel.load_from_checkpoint(model_name, best=True)\n            ..\n\n        If ``file_name`` is given, returns the model saved under\n        \'{work_dir}/darts_logs/{model_name}/checkpoints/{file_name}\'.\n\n        If ``file_name`` is not given, will try to restore the best checkpoint (if ``best`` is ``True``) or the most\n        recent checkpoint (if ``best`` is ``False`` from \'{work_dir}/darts_logs/{model_name}/checkpoints/\'.\n\n        Example for loading an :class:`RNNModel` checkpoint to CPU that was saved on GPU:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from darts.models import RNNModel\n\n                model_loaded = RNNModel.load_from_checkpoint(model_name, best=True, map_location="cpu")\n                model_loaded.to_cpu()\n            ..\n\n        Parameters\n        ----------\n        model_name\n            The name of the model, used to retrieve the checkpoints folder\'s name.\n        work_dir\n            Working directory (containing the checkpoints folder). Defaults to current working directory.\n        file_name\n            The name of the checkpoint file. If not specified, use the most recent one.\n        best\n            If set, will retrieve the best model (according to validation loss) instead of the most recent one. Only\n            is ignored when ``file_name`` is given.\n        **kwargs\n            Additional kwargs for PyTorch Lightning\'s :func:`LightningModule.load_from_checkpoint()` method,\n            such as ``map_location`` to load the model onto a different device than the one from which it was saved.\n            For more information, read the `official documentation <https://pytorch-lightning.readthedocs.io/en/stable/\n            common/lightning_module.html#load-from-checkpoint>`_.\n\n\n        Returns\n        -------\n        TorchForecastingModel\n            The corresponding trained :class:`TorchForecastingModel`.\n        '
        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)
        checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
        model_dir = _get_runs_folder(work_dir, model_name)
        base_model_path = os.path.join(model_dir, INIT_MODEL_NAME)
        raise_if_not(os.path.exists(base_model_path), f'Could not find base model save file `{INIT_MODEL_NAME}` in {model_dir}.', logger)
        model: TorchForecastingModel = torch.load(base_model_path, map_location=kwargs.get('map_location'))
        if file_name is None:
            file_name = _get_checkpoint_fname(work_dir, model_name, best=best)
        file_path = os.path.join(checkpoint_dir, file_name)
        logger.info(f'loading {file_name}')
        model.model = model._load_from_checkpoint(file_path, **kwargs)
        loss_fn = model.model_params.get('loss_fn')
        if loss_fn is not None:
            model.model.criterion = loss_fn
        torch_metrics = model.model.configure_torch_metrics(model.model_params.get('torch_metrics'))
        model.model.train_metrics = torch_metrics.clone(prefix='train_')
        model.model.val_metrics = torch_metrics.clone(prefix='val_')
        model._fit_called = True
        model.load_ckpt_path = file_path
        return model

    def _load_from_checkpoint(self, file_path, **kwargs):
        if False:
            i = 10
            return i + 15
        'Loads a checkpoint for the underlying :class:`PLForecastingModule` (PLM) model.\n        The PLM object is not stored when saving a :class:`TorchForecastingModel` (TFM) to avoid saving\n        the model twice. Instead, we recover the module class with the module path and class name stored\n        in the TFM object. With the recovered module class, we can load the checkpoint.\n        '
        pl_module_cls = getattr(sys.modules[self._module_path], self._module_name)
        return pl_module_cls.load_from_checkpoint(file_path, **kwargs)

    def load_weights_from_checkpoint(self, model_name: str=None, work_dir: str=None, file_name: str=None, best: bool=True, strict: bool=True, load_encoders: bool=True, skip_checks: bool=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Load only the weights from automatically saved checkpoints under '{work_dir}/darts_logs/{model_name}/\n        checkpoints/'. This method is used for models that were created with ``save_checkpoints=True`` and\n        that need to be re-trained or fine-tuned with different optimizer or learning rate scheduler. However,\n        it can also be used to load weights for inference.\n\n        To resume an interrupted training, please consider using :meth:`load_from_checkpoint()\n        <TorchForecastingModel.load_from_checkpoint()>` which also reload the trainer, optimizer and\n        learning rate scheduler states.\n\n        For manually saved model, consider using :meth:`load() <TorchForecastingModel.load()>` or\n        :meth:`load_weights() <TorchForecastingModel.load_weights()>` instead.\n\n        Note: This method needs to be able to access the darts model checkpoint (.pt) in order to load the encoders\n        and perform sanity checks on the model parameters.\n\n        Parameters\n        ----------\n        model_name\n            The name of the model, used to retrieve the checkpoints folder's name. Default: ``self.model_name``.\n        work_dir\n            Working directory (containing the checkpoints folder). Defaults to current working directory.\n        file_name\n            The name of the checkpoint file. If not specified, use the most recent one.\n        best\n            If set, will retrieve the best model (according to validation loss) instead of the most recent one. Only\n            is ignored when ``file_name`` is given. Default: ``True``.\n        strict\n            If set, strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict().\n            Default: ``True``.\n            For more information, read the `official documentation <https://pytorch.org/docs/stable/generated/torch.\n            nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict>`_.\n        load_encoders\n            If set, will load the encoders from the model to enable direct call of fit() or predict().\n            Default: ``True``.\n        skip_checks\n            If set, will disable the loading of the encoders and the sanity checks on model parameters\n            (not recommended). Cannot be used with `load_encoders=True`. Default: ``False``.\n        **kwargs\n            Additional kwargs for PyTorch's :func:`load` method, such as ``map_location`` to load the model onto a\n            different device than the one from which it was saved.\n            For more information, read the `official documentation <https://pytorch.org/docs/stable/generated/\n            torch.load.html>`_.\n        "
        raise_if('weights_only' in kwargs.keys() and kwargs['weights_only'], 'Passing `weights_only=True` to `torch.load` will disrupt this method sanity checks.', logger)
        raise_if(skip_checks and load_encoders, '`skip-checks` and `load_encoders` are mutually exclusive parameters and cannot be both set to `True`.', logger)
        if model_name is None:
            model_name = self.model_name
        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)
        if file_name is None:
            file_name = _get_checkpoint_fname(work_dir, model_name, best=best)
        if file_name[:5] == 'last-' or file_name[:5] == 'best-':
            checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
            tfm_save_file_dir = _get_runs_folder(work_dir, model_name)
            tfm_save_file_name = INIT_MODEL_NAME
        else:
            checkpoint_dir = ''
            tfm_save_file_dir = checkpoint_dir
            tfm_save_file_name = file_name[:-5]
        ckpt_path = os.path.join(checkpoint_dir, file_name)
        ckpt = torch.load(ckpt_path, **kwargs)
        raise_if_not('train_sample_shape' in ckpt.keys(), "The provided checkpoint was generated with darts release <= 0.23.1 and it is missing the 'train_sample_shape' key. This value must be computed from the `model.train_sample` attribute and manually added to the checkpoint prior to loading.", logger)
        np_dtype = TORCH_NP_DTYPES[ckpt['model_dtype']]
        mock_train_sample = [np.zeros(sample_shape, dtype=np_dtype) if sample_shape else None for sample_shape in ckpt['train_sample_shape']]
        self.train_sample = tuple(mock_train_sample)
        if not skip_checks:
            tfm_save_file_path = os.path.join(tfm_save_file_dir, tfm_save_file_name)
            if not os.path.exists(tfm_save_file_path):
                raise_log(FileNotFoundError(f'Could not find {tfm_save_file_path}, necessary to load the encoders and run sanity checks on the model parameters.'), logger)
            with open(tfm_save_file_path, 'rb') as tfm_save_file:
                tfm_save: TorchForecastingModel = torch.load(tfm_save_file, map_location=kwargs.get('map_location', None))
            (self.encoders, self.add_encoders) = self._load_encoders(tfm_save, load_encoders)
            self._check_ckpt_parameters(tfm_save)
        self.model = self._init_model()
        self.model.to_dtype(ckpt['model_dtype'])
        self.model.load_state_dict(ckpt['state_dict'], strict=strict)
        self._fit_called = True

    def load_weights(self, path: str, load_encoders: bool=True, skip_checks: bool=False, **kwargs):
        if False:
            return 10
        '\n        Loads the weights from a manually saved model (saved with :meth:`save() <TorchForecastingModel.save()>`).\n\n        Note: This method needs to be able to access the darts model checkpoint (.pt) in order to load the encoders\n        and perform sanity checks on the model parameters.\n\n        Parameters\n        ----------\n        path\n            Path from which to load the model\'s weights. If no path was specified when saving the model, the\n            automatically generated path ending with ".pt" has to be provided.\n        load_encoders\n            If set, will load the encoders from the model to enable direct call of fit() or predict().\n            Default: ``True``.\n        skip_checks\n            If set, will disable the loading of the encoders and the sanity checks on model parameters\n            (not recommended). Cannot be used with `load_encoders=True`. Default: ``False``.\n        **kwargs\n            Additional kwargs for PyTorch\'s :func:`load` method, such as ``map_location`` to load the model onto a\n            different device than the one from which it was saved.\n            For more information, read the `official documentation <https://pytorch.org/docs/stable/generated/\n            torch.load.html>`_.\n\n        '
        path_ptl_ckpt = path + '.ckpt'
        raise_if_not(os.path.exists(path_ptl_ckpt), f'Could not find PyTorch LightningModule checkpoint {path_ptl_ckpt}.', logger)
        self.load_weights_from_checkpoint(file_name=path_ptl_ckpt, load_encoders=load_encoders, skip_checks=skip_checks, **kwargs)

    def to_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Updates the PyTorch Lightning Trainer parameters to move the model to CPU the next time :fun:`fit()` or\n        :func:`predict()` is called.\n        '
        self.trainer_params['accelerator'] = 'cpu'
        self.trainer_params = {k: v for (k, v) in self.trainer_params.items() if k not in ['devices', 'auto_select_gpus']}

    @property
    def model_created(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.model is not None

    @property
    def epochs_trained(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.model.epochs_trained if self.model_created else 0

    @property
    def likelihood(self) -> Optional[Likelihood]:
        if False:
            for i in range(10):
                print('nop')
        return self.model.likelihood if self.model_created else self.pl_module_params.get('likelihood', None)

    @property
    def input_chunk_length(self) -> int:
        if False:
            print('Hello World!')
        return self.model.input_chunk_length if self.model_created else self.pl_module_params['input_chunk_length']

    @property
    def output_chunk_length(self) -> int:
        if False:
            return 10
        return self.model.output_chunk_length if self.model_created else self.pl_module_params['output_chunk_length']

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.model._is_probabilistic if self.model_created else True

    def _check_optimizable_historical_forecasts(self, forecast_horizon: int, retrain: Union[bool, int, Callable[..., bool]], show_warnings: bool) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Historical forecast can be optimized only if `retrain=False` and `forecast_horizon <= model.output_chunk_length`\n        (no auto-regression required).\n        '
        return _check_optimizable_historical_forecasts_global_models(model=self, forecast_horizon=forecast_horizon, retrain=retrain, show_warnings=show_warnings, allow_autoregression=True)

    def _optimized_historical_forecasts(self, series: Optional[Sequence[TimeSeries]], past_covariates: Optional[Sequence[TimeSeries]]=None, future_covariates: Optional[Sequence[TimeSeries]]=None, num_samples: int=1, start: Optional[Union[pd.Timestamp, float, int]]=None, start_format: Literal['position', 'value']='value', forecast_horizon: int=1, stride: int=1, overlap_end: bool=False, last_points_only: bool=True, verbose: bool=False, show_warnings: bool=True, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]]:
        if False:
            while True:
                i = 10
        '\n        For TorchForecastingModels we use a strided inference dataset to avoid having to recreate trainers and\n        datasets for each forecastable index and series.\n        '
        (series, past_covariates, future_covariates) = _process_historical_forecast_input(model=self, series=series, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, allow_autoregression=True)
        forecasts_list = _optimized_historical_forecasts(model=self, series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=num_samples, start=start, start_format=start_format, forecast_horizon=forecast_horizon, stride=stride, overlap_end=overlap_end, last_points_only=last_points_only, show_warnings=show_warnings, predict_likelihood_parameters=predict_likelihood_parameters, verbose=verbose)
        return forecasts_list

    def _load_encoders(self, tfm_save: 'TorchForecastingModel', load_encoders: bool) -> Tuple[SequentialEncoder, Dict]:
        if False:
            return 10
        'Return the encoders from a model save with several sanity checks.'
        if self.add_encoders is None:
            same_encoders = True
            same_transformer = True
        elif tfm_save.add_encoders is None:
            same_encoders = False
            same_transformer = False
        else:
            self_transformer = self.add_encoders.get('transformer', None)
            tfm_transformer = tfm_save.add_encoders.get('transformer', None)
            same_transformer = type(self_transformer) == type(tfm_transformer)
            self_encoders = {k: v for (k, v) in self.add_encoders.items() if k != 'transformer'}
            tfm_encoders = {k: v for (k, v) in tfm_save.add_encoders.items() if k != 'transformer'}
            same_encoders = self_encoders == tfm_encoders
        if load_encoders:
            raise_if_not(same_transformer, f"Transformers defined in the loaded encoders and the new model must have the same type, received ({(None if tfm_save.add_encoders is None else type(tfm_save.add_encoders.get('transformer', None)))}) and ({(None if self.add_encoders is None else type(self.add_encoders.get('transformer', None)))}).", logger)
            raise_if_not(same_encoders, f'Encoders loaded from the checkpoint ({tfm_save.add_encoders}) are different from the encoders defined in the new model ({self.add_encoders}).', logger)
            new_add_encoders: Dict = copy.deepcopy(tfm_save.add_encoders)
            new_encoders: SequentialEncoder = copy.deepcopy(tfm_save.encoders)
        else:
            raise_if(len(tfm_save.add_encoders) > 0 and self.add_encoders is None, f'Model was created without encoders and encoders were not loaded, but the weights were trained using encoders({tfm_save.add_encoders}). Either set `load_encoders` to `True` or add a matching `add_encoders` dict at model creation.', logger)
            new_add_encoders: Dict = self.add_encoders
            new_encoders: SequentialEncoder = self.initialize_encoders()
            if tfm_save.encoders is not None:
                (ckpt_past_enc_n_comp, ckpt_future_enc_n_comp) = tfm_save.encoders.encoding_n_components
                (new_past_enc_n_comp, new_future_enc_n_comp) = new_encoders.encoding_n_components
                raise_if(new_past_enc_n_comp != ckpt_past_enc_n_comp or new_future_enc_n_comp != ckpt_future_enc_n_comp, f"Number of components mismatch between model's and checkpoint's encoders:\n- past covs: new {new_past_enc_n_comp}, checkpoint {ckpt_past_enc_n_comp}\n- future covs: new {new_future_enc_n_comp}, checkpoint {ckpt_future_enc_n_comp}", logger)
                if not new_encoders.fit_called and new_encoders.requires_fit:
                    logger.info("Model's weights were loaded without the encoders and at least one of them needs to be fitted: please call `fit()` before calling `predict()`.")
        return (new_encoders, new_add_encoders)

    def _check_ckpt_parameters(self, tfm_save):
        if False:
            while True:
                i = 10
        '\n        Check that the positional parameters used to instantiate the new model loading the weights match those\n        of the saved model, to return meaningful messages in case of discrepancies.\n        '
        skipped_params = list(inspect.signature(TorchForecastingModel.__init__).parameters.keys()) + ['loss_fn', 'torch_metrics', 'optimizer_cls', 'optimizer_kwargs', 'lr_scheduler_cls', 'lr_scheduler_kwargs']
        params_to_check = set(tfm_save.model_params.keys()).union(self.model_params.keys()) - set(skipped_params)
        incorrect_params = []
        missing_params = []
        for param_key in params_to_check:
            if param_key not in self.model_params.keys():
                missing_params.append((param_key, tfm_save.model_params[param_key]))
            elif param_key not in tfm_save.model_params.keys():
                incorrect_params.append((param_key, None, self.model_params[param_key]))
            elif self.model_params[param_key] != tfm_save.model_params[param_key]:
                incorrect_params.append((param_key, tfm_save.model_params[param_key], self.model_params[param_key]))
        if len(missing_params) + len(incorrect_params) > 0:
            msg = ['The values of the hyper-parameters in the model and loaded checkpoint should be identical.']
            if len(missing_params) > 0:
                msg += ['missing :']
                msg += [f'   - {param}={exp_val}' for (param, exp_val) in missing_params]
            if len(incorrect_params) > 0:
                msg += ['incorrect :']
                msg += [f'   - found {param}={cur_val}, should be {param}={exp_val}' for (param, exp_val, cur_val) in incorrect_params]
            raise_log(ValueError('\n'.join(msg)), logger)

    def __getstate__(self):
        if False:
            return 10
        return {k: v for (k, v) in self.__dict__.items() if k not in TFM_ATTRS_NO_PICKLE}

    def __setstate__(self, d):
        if False:
            return 10
        self.__dict__ = d
        for (attr, default_val) in TFM_ATTRS_NO_PICKLE.items():
            setattr(self, attr, default_val)

def _raise_if_wrong_type(obj, exp_type, msg='expected type {}, got: {}'):
    if False:
        for i in range(10):
            print('nop')
    raise_if_not(isinstance(obj, exp_type), msg.format(exp_type, type(obj)))
'\nBelow we define the 5 torch model types:\n    * PastCovariatesTorchModel\n    * FutureCovariatesTorchModel\n    * DualCovariatesTorchModel\n    * MixedCovariatesTorchModel\n    * SplitCovariatesTorchModel\n'

def _basic_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    if False:
        print('Hello World!')
    '\n    For all models relying on one type of covariates only (Past, Future, Dual), we can rely on the fact\n    that training/inference datasets have target and covariates in first and second position to do the checks.\n\n    - `train_sample` comes with last dimension (static covs, target TimeSeries)\n    - `predict_sample` comes with last dimensions (..., static covs, target TimeSeries, first prediction time stamp)\n\n    '
    (tgt_train, cov_train, static_train) = train_sample[:2] + (train_sample[-2],)
    (tgt_pred, cov_pred, static_pred) = predict_sample[:2] + (predict_sample[-3],)
    raise_if_not(tgt_train.shape[-1] == tgt_pred.shape[-1], 'The provided target has a dimension (width) that does not match the dimension of the target this model has been trained on.')
    for ((c_train, c_pred), c_descr) in zip([(cov_train, cov_pred), (static_train, static_pred)], ['past or future covariates', 'static covariates']):
        raise_if(c_train is not None and c_pred is None, f'This model has been trained with {c_descr}; covariates of matching dimensionality are required for prediction.')
        raise_if(c_train is None and c_pred is not None, f'This model has been trained without {c_descr}. No {c_descr} should be provided for prediction.')
        raise_if(c_train is not None and c_pred is not None and (c_train.shape[-1] != c_pred.shape[-1] if c_descr != 'static covariates' else c_train.shape != c_pred.shape), f'The provided {c_descr} must have dimensionality matching that of the covariates used for training the model.')

def _mixed_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    if False:
        while True:
            i = 10
    '\n    For models relying on MixedCovariates.\n\n    Parameters\n    ----------\n    train_sample\n        (past_target, past_covariates, historic_future_covariates, future_covariates, future_target)\n    predict_sample\n        (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates, ts_target)\n    '
    ds_names = ['past_target', 'past_covariates', 'historic_future_covariates', 'future_covariates', 'static_covariates']
    train_has_ds = [ds is not None for ds in train_sample[:-1]]
    predict_has_ds = [ds is not None for ds in predict_sample[:4] + (predict_sample[5],)]
    train_datasets = train_sample[:-1]
    predict_datasets = predict_sample[:4] + (predict_sample[5],)
    (tgt_train, tgt_pred) = (train_datasets[0], predict_datasets[0])
    raise_if_not(tgt_train.shape[-1] == tgt_pred.shape[-1], 'The provided target has a dimension (width) that does not match the dimension of the target this model has been trained on.')
    for (idx, (ds_in_train, ds_in_predict, ds_name)) in enumerate(zip(train_has_ds, predict_has_ds, ds_names)):
        raise_if(ds_in_train and (not ds_in_predict), f'This model has been trained with `{ds_name}`; some `{ds_name}` of matching dimensionality are needed for prediction.')
        raise_if(not ds_in_train and ds_in_predict, f'This model has been trained without `{ds_name}`; No `{ds_name}` should be provided for prediction.')
        raise_if(ds_in_train and ds_in_predict and (train_datasets[idx].shape[-1] != predict_datasets[idx].shape[-1] if ds_name != 'static_covariates' else train_datasets[idx].shape != predict_datasets[idx].shape), f'The provided `{ds_name}` must have equal dimensionality as the `{ds_name}` used for training the model.')

class PastCovariatesTorchModel(TorchForecastingModel, ABC):
    supports_future_covariates = False

    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> PastCovariatesTrainingDataset:
        if False:
            print('Hello World!')
        raise_if_not(future_covariates is None, 'Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).')
        return PastCovariatesSequentialDataset(target_series=target, covariates=past_covariates, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=self.uses_static_covariates)

    def _build_inference_dataset(self, target: Sequence[TimeSeries], n: int, past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], stride: int=0, bounds: Optional[np.ndarray]=None) -> PastCovariatesInferenceDataset:
        if False:
            for i in range(10):
                print('nop')
        raise_if_not(future_covariates is None, 'Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).')
        return PastCovariatesInferenceDataset(target_series=target, covariates=past_covariates, n=n, stride=stride, bounds=bounds, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, use_static_covariates=self.uses_static_covariates)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if False:
            for i in range(10):
                print('nop')
        _raise_if_wrong_type(train_dataset, PastCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        if False:
            i = 10
            return i + 15
        _raise_if_wrong_type(inference_dataset, PastCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        if False:
            print('Hello World!')
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        if False:
            while True:
                i = 10
        raise_if_not(future_covariates is None, 'Some future_covariates have been provided to a PastCovariates model. These models support only past_covariates.')

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            return 10
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = False
        return (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates, None, None)

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            for i in range(10):
                print('nop')
        return (-self.input_chunk_length, self.output_chunk_length - 1, -self.input_chunk_length if self.uses_past_covariates else None, -1 if self.uses_past_covariates else None, None, None)

class FutureCovariatesTorchModel(TorchForecastingModel, ABC):
    supports_past_covariates = False

    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> FutureCovariatesTrainingDataset:
        if False:
            i = 10
            return i + 15
        raise_if_not(past_covariates is None, 'Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).')
        return FutureCovariatesSequentialDataset(target_series=target, covariates=future_covariates, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=self.uses_static_covariates)

    def _build_inference_dataset(self, target: Sequence[TimeSeries], n: int, past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], stride: int=0, bounds: Optional[np.ndarray]=None) -> FutureCovariatesInferenceDataset:
        if False:
            return 10
        raise_if_not(past_covariates is None, 'Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).')
        return FutureCovariatesInferenceDataset(target_series=target, covariates=future_covariates, n=n, stride=stride, bounds=bounds, input_chunk_length=self.input_chunk_length, use_static_covariates=self.uses_static_covariates)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if False:
            i = 10
            return i + 15
        _raise_if_wrong_type(train_dataset, FutureCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        if False:
            return 10
        _raise_if_wrong_type(inference_dataset, FutureCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        if False:
            print('Hello World!')
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        if False:
            while True:
                i = 10
        raise_if_not(past_covariates is None, 'Some past_covariates have been provided to a PastCovariates model. These models support only future_covariates.')

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            return 10
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = False
        takes_future_covariates = True
        return (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates, None, None)

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            for i in range(10):
                print('nop')
        return (-self.input_chunk_length, self.output_chunk_length - 1, None, None, 0 if self.uses_future_covariates else None, self.output_chunk_length - 1 if self.uses_future_covariates else None)

class DualCovariatesTorchModel(TorchForecastingModel, ABC):
    supports_past_covariates = False

    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> DualCovariatesTrainingDataset:
        if False:
            print('Hello World!')
        return DualCovariatesSequentialDataset(target_series=target, covariates=future_covariates, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=self.uses_static_covariates)

    def _build_inference_dataset(self, target: Sequence[TimeSeries], n: int, past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], stride: int=0, bounds: Optional[np.ndarray]=None) -> DualCovariatesInferenceDataset:
        if False:
            i = 10
            return i + 15
        return DualCovariatesInferenceDataset(target_series=target, covariates=future_covariates, n=n, stride=stride, bounds=bounds, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, use_static_covariates=self.uses_static_covariates)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if False:
            print('Hello World!')
        _raise_if_wrong_type(train_dataset, DualCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        if False:
            print('Hello World!')
        _raise_if_wrong_type(inference_dataset, DualCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        if False:
            while True:
                i = 10
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        if False:
            print('Hello World!')
        raise_if_not(past_covariates is None, 'Some past_covariates have been provided to a DualCovariates Torch model. These models support only future_covariates.')

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            for i in range(10):
                print('nop')
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = False
        takes_future_covariates = True
        return (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates, None, None)

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            for i in range(10):
                print('nop')
        return (-self.input_chunk_length, self.output_chunk_length - 1, None, None, -self.input_chunk_length if self.uses_future_covariates else None, self.output_chunk_length - 1 if self.uses_future_covariates else None)

class MixedCovariatesTorchModel(TorchForecastingModel, ABC):

    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> MixedCovariatesTrainingDataset:
        if False:
            while True:
                i = 10
        return MixedCovariatesSequentialDataset(target_series=target, past_covariates=past_covariates, future_covariates=future_covariates, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=self.uses_static_covariates)

    def _build_inference_dataset(self, target: Sequence[TimeSeries], n: int, past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], stride: int=0, bounds: Optional[np.ndarray]=None) -> MixedCovariatesInferenceDataset:
        if False:
            for i in range(10):
                print('nop')
        return MixedCovariatesInferenceDataset(target_series=target, past_covariates=past_covariates, future_covariates=future_covariates, n=n, stride=stride, bounds=bounds, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, use_static_covariates=self.uses_static_covariates)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if False:
            i = 10
            return i + 15
        _raise_if_wrong_type(train_dataset, MixedCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        if False:
            for i in range(10):
                print('nop')
        _raise_if_wrong_type(inference_dataset, MixedCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        if False:
            for i in range(10):
                print('nop')
        _mixed_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        if False:
            return 10
        pass

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            print('Hello World!')
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates, None, None)

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            i = 10
            return i + 15
        return (-self.input_chunk_length, self.output_chunk_length - 1, -self.input_chunk_length if self.uses_past_covariates else None, -1 if self.uses_past_covariates else None, -self.input_chunk_length if self.uses_future_covariates else None, self.output_chunk_length - 1 if self.uses_future_covariates else None)

class SplitCovariatesTorchModel(TorchForecastingModel, ABC):

    def _build_train_dataset(self, target: Sequence[TimeSeries], past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], max_samples_per_ts: Optional[int]) -> SplitCovariatesTrainingDataset:
        if False:
            i = 10
            return i + 15
        return SplitCovariatesSequentialDataset(target_series=target, past_covariates=past_covariates, future_covariates=future_covariates, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, max_samples_per_ts=max_samples_per_ts, use_static_covariates=self.uses_static_covariates)

    def _build_inference_dataset(self, target: Sequence[TimeSeries], n: int, past_covariates: Optional[Sequence[TimeSeries]], future_covariates: Optional[Sequence[TimeSeries]], stride: int=0, bounds: Optional[np.ndarray]=None) -> SplitCovariatesInferenceDataset:
        if False:
            while True:
                i = 10
        return SplitCovariatesInferenceDataset(target_series=target, past_covariates=past_covariates, future_covariates=future_covariates, n=n, stride=stride, bounds=bounds, input_chunk_length=self.input_chunk_length, output_chunk_length=self.output_chunk_length, use_static_covariates=self.uses_static_covariates)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if False:
            return 10
        _raise_if_wrong_type(train_dataset, SplitCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        if False:
            i = 10
            return i + 15
        _raise_if_wrong_type(inference_dataset, SplitCovariatesInferenceDataset)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        if False:
            i = 10
            return i + 15
        pass

    def _verify_predict_sample(self, predict_sample: Tuple):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            return 10
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates, None, None)

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            while True:
                i = 10
        return (-self.input_chunk_length, self.output_chunk_length - 1, -self.input_chunk_length if self.uses_past_covariates else None, -1 if self.uses_past_covariates else None, 0 if self.uses_future_covariates else None, self.output_chunk_length - 1 if self.uses_future_covariates else None)