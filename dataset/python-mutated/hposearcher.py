from typing import Any
import pytorch_lightning as pl
import copy
import math
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.automl.hpo.backend import create_hpo_backend, SamplerType
from .objective import Objective
from ._helper import ResetCallback, CustomEvaluationLoop
from bigdl.nano.automl.utils.parallel import run_parallel
from bigdl.nano.automl.hpo.search import _search_summary, _end_search, _create_study, _validate_args, _prepare_args

class HPOSearcher:
    """Hyper Parameter Searcher. A Tuner-like class."""
    FIT_KEYS = {'train_dataloaders', 'val_dataloaders', 'datamodule', 'ckpt_path'}
    EXTRA_FIT_KEYS = {'max_epochs'}
    TUNE_CREATE_KEYS = {'storage', 'sampler', 'sampler_kwargs', 'pruner', 'pruner_kwargs', 'study_name', 'load_if_exists', 'direction', 'directions'}
    TUNE_RUN_KEYS = {'n_trials', 'timeout', 'n_jobs', 'catch', 'tune_callbacks', 'gc_after_trial', 'show_progress_bar'}

    def __init__(self, trainer: 'pl.Trainer', num_processes: int=1) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Init a HPO Searcher.\n\n        :param trainer: The pl.Trainer object.\n        '
        self.trainer = trainer
        self.num_process = num_processes
        if num_processes == 1:
            callbacks = self.trainer.callbacks or []
            callbacks.append(ResetCallback())
            self.trainer.callbacks = callbacks
        self.model_class = pl.LightningModule
        self.study = None
        self.objective = None
        self.tune_end = False
        self._lazymodel = None
        self.backend = create_hpo_backend()
        self.create_kwargs = None
        self.run_kwargs = None
        self.fit_kwargs = None

    def _create_objective(self, model, target_metric, mode, create_kwargs, acceleration, input_sample, fit_kwargs):
        if False:
            for i in range(10):
                print('nop')
        isprune = True if create_kwargs.get('pruner', None) else False
        direction = create_kwargs.get('direction', None)
        directions = create_kwargs.get('directions', None)
        self.objective = Objective(searcher=self, model=model._model_build, target_metric=target_metric, mode=mode, pruning=isprune, direction=direction, directions=directions, acceleration=acceleration, input_sample=input_sample, **fit_kwargs)

    def _run_search(self):
        if False:
            for i in range(10):
                print('nop')
        self.trainer.state.fn = TrainerFn.TUNING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.tuning = True
        self.study.optimize(self.objective, **self.run_kwargs)
        self.tune_end = False
        self.trainer.tuning = False
        self.trainer.state.status = TrainerStatus.FINISHED
        invalidInputError(self.trainer.state.stopped, 'trainer state should be stopped')

    def _run_search_n_procs(self, n_procs=4):
        if False:
            print('Hello World!')
        new_searcher = copy.deepcopy(self)
        n_trials = new_searcher.run_kwargs.get('n_trials', None)
        if n_trials:
            subp_n_trials = math.ceil(n_trials / n_procs)
            new_searcher.run_kwargs['n_trials'] = subp_n_trials
        run_parallel(func=new_searcher._run_search, kwargs={}, n_procs=n_procs)

    def search(self, model, resume=False, target_metric=None, mode='best', n_parallels=1, acceleration=False, input_sample=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Run HPO Searcher. It will be called in Trainer.search().\n\n        :param model: The model to be searched. It should be an automodel.\n        :param resume: whether to resume the previous or start a new one,\n            defaults to False.\n        :param target_metric: the object metric to optimize,\n            defaults to None.\n        :param mode: use last epoch\'s result as trial\'s score or use best epoch\'s.\n            defaults to \'best\', you can change it to \'last\'.\n        :param acceleration: Whether to automatically consider the model after\n            inference acceleration in the search process. It will only take\n            effect if target_metric contains "latency". Default value is False.\n        :param input_sample: A set of inputs for trace, defaults to None if you have\n            trace before or model is a LightningModule with any dataloader attached.\n        :param return: the model with study meta info attached.\n        '
        search_kwargs = kwargs or {}
        self.target_metric = target_metric
        _validate_args(search_kwargs, self.target_metric, legal_keys=[HPOSearcher.FIT_KEYS, HPOSearcher.EXTRA_FIT_KEYS, HPOSearcher.TUNE_CREATE_KEYS, HPOSearcher.TUNE_RUN_KEYS])
        _sampler_kwargs = model._lazyobj.sampler_kwargs
        user_sampler_kwargs = kwargs.get('sampler_kwargs', {})
        _sampler_kwargs.update(user_sampler_kwargs)
        if 'sampler' in kwargs and kwargs['sampler'] in [SamplerType.Grid]:
            search_kwargs['sampler_kwargs'] = _sampler_kwargs
            invalidInputError(len(model._lazyobj.kwspaces_) <= len(_sampler_kwargs), 'Only `space.Categorical` is supported for `SamplerType.Grid` sampler. Please try replace other space to `space.Categorical` or use another SamplerType.')
        (self.create_kwargs, self.run_kwargs, self.fit_kwargs) = _prepare_args(search_kwargs, HPOSearcher.TUNE_CREATE_KEYS, HPOSearcher.TUNE_RUN_KEYS, HPOSearcher.FIT_KEYS, self.backend)
        if self.study is None:
            self.study = _create_study(resume, self.create_kwargs, self.backend)
        if self.objective is None:
            self._create_objective(model, self.target_metric, mode, self.create_kwargs, acceleration, input_sample, self.fit_kwargs)
        if n_parallels and n_parallels > 1:
            invalidInputError(self.create_kwargs.get('storage', '').strip() != '', 'parallel search is not supported when in-mem storage is used (n_parallels must be 1)')
            self._run_search_n_procs(n_procs=n_parallels)
        else:
            self._run_search()
        if not self.objective.mo_hpo:
            self._lazymodel = _end_search(study=self.study, model_builder=model._model_build, use_trial_id=-1)
            return self._lazymodel
        else:
            return self.study.best_trials

    def search_summary(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrive a summary of trials.\n\n        :return: A summary of all the trials. Currently the entire study is\n            returned to allow more flexibility for further analysis and visualization.\n        '
        return _search_summary(self.study)

    def end_search(self, use_trial_id=-1):
        if False:
            i = 10
            return i + 15
        '\n        Put an end to tuning.\n\n        Use the specified trial or best trial to init and build the model.\n\n        :param use_trial_id: int(optional) params of which trial to be used.\n            Defaults to -1.\n        :throw: ValueError: error when tune is not called before end_search.\n        '
        self._lazymodel = _end_search(study=self.study, model_builder=self._model_build, use_trial_id=use_trial_id)
        self.tune_end = True
        return self._lazymodel

    def _run(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        '`_run` wrapper to set the proper state during tuning,        as this can be called multiple times.'
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.training = True
        self.trainer.state.fn = TrainerFn.FITTING
        if self.num_process > 1:
            self.trainer.fit(*args, **kwargs)
        else:
            self.trainer._run(*args, **kwargs)
        self.trainer.tuning = True

    def _validate(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'A wrapper to test optimization latency multiple times after training.'
        self.trainer.validate_loop = CustomEvaluationLoop()
        self.trainer.state.fn = TrainerFn.VALIDATING
        self.trainer.training = False
        self.trainer.testing = False
        self.trainer.validate(*args, **kwargs)