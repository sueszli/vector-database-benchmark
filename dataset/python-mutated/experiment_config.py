"""
Top level experiement configuration class, ``ExperimentConfig``.
"""
__all__ = ['ExperimentConfig']
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
from typing_extensions import Literal
import yaml
from .algorithm import _AlgorithmConfig
from .base import ConfigBase
from .shared_storage import SharedStorageConfig
from .training_service import TrainingServiceConfig
from . import utils

@dataclass(init=False)
class ExperimentConfig(ConfigBase):
    """
    Class of experiment configuration. Check the reference_ for explaination of each field.

    When used in Python experiment API, it can be constructed in two favors:

    1. Create an empty project then set each field

    .. code-block:: python

        config = ExperimentConfig('local')
        config.search_space = {...}
        config.tuner.name = 'random'
        config.training_service.use_active_gpu = True

     2. Use kwargs directly

     .. code-block:: python

        config = ExperimentConfig(
            search_space = {...},
            tuner = AlgorithmConfig(name='random'),
            training_service = LocalConfig(
                use_active_gpu = True
            )
        )

    .. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html
    """
    experiment_name: Optional[str] = None
    experiment_type: Literal['hpo'] = 'hpo'
    search_space_file: Optional[utils.PathLike] = None
    search_space: Any = None
    trial_command: Optional[str] = None
    trial_code_directory: utils.PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: Optional[int] = None
    max_experiment_duration: Union[str, int, None] = None
    max_trial_number: Optional[int] = None
    max_trial_duration: Union[str, int, None] = None
    nni_manager_ip: Optional[str] = None
    use_annotation: bool = False
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: utils.PathLike = '~/nni-experiments'
    tuner_gpu_indices: Union[List[int], int, str, None] = None
    tuner: Optional[_AlgorithmConfig] = None
    assessor: Optional[_AlgorithmConfig] = None
    advisor: Optional[_AlgorithmConfig] = None
    training_service: Union[TrainingServiceConfig, List[TrainingServiceConfig]]
    shared_storage: Optional[SharedStorageConfig] = None

    def __new__(cls, *args, **kwargs) -> 'ExperimentConfig':
        if False:
            print('Hello World!')
        if cls is not ExperimentConfig:
            return super().__new__(cls)
        if kwargs.get('experimentType') == 'nas':
            from nni.nas.experiment import NasExperimentConfig
            return NasExperimentConfig.__new__(NasExperimentConfig)
        else:
            return super().__new__(cls)

    def __init__(self, training_service_platform=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if training_service_platform is not None:
            assert utils.is_missing(self.training_service)
            if isinstance(training_service_platform, list):
                self.training_service = [utils.training_service_config_factory(ts) for ts in training_service_platform]
            else:
                self.training_service = utils.training_service_config_factory(training_service_platform)
            for algo_type in ['tuner', 'assessor', 'advisor']:
                if getattr(self, algo_type) is None:
                    setattr(self, algo_type, _AlgorithmConfig(name='_none_', class_args={}))
        elif not utils.is_missing(self.training_service):
            if isinstance(self.training_service, list):
                self.training_service = [utils.load_training_service_config(ts) for ts in self.training_service]
            else:
                self.training_service = utils.load_training_service_config(self.training_service)

    def _canonicalize(self, _parents):
        if False:
            for i in range(10):
                print('nop')
        if self.log_level is None:
            self.log_level = 'debug' if self.debug else 'info'
        self.tuner_gpu_indices = utils.canonical_gpu_indices(self.tuner_gpu_indices)
        for algo_type in ['tuner', 'assessor', 'advisor']:
            algo = getattr(self, algo_type)
            if isinstance(algo, dict):
                _AlgorithmConfig(**algo)
            if algo is not None and algo.name == '_none_':
                setattr(self, algo_type, None)
        if self.advisor is not None:
            assert self.tuner is None, '"advisor" is deprecated. You should only set "tuner".'
            self.tuner = self.advisor
            self.advisor = None
        super()._canonicalize([self])
        if self.search_space_file is not None:
            yaml_error = None
            try:
                self.search_space = _load_search_space_file(self.search_space_file)
            except Exception as e:
                yaml_error = repr(e)
            if yaml_error is not None:
                msg = f'ExperimentConfig: Failed to load search space file "{self.search_space_file}": {yaml_error}'
                raise ValueError(msg)
        if self.nni_manager_ip is None:
            platform = getattr(self.training_service, 'platform')
            has_ip = isinstance(getattr(self.training_service, 'nni_manager_ip'), str)
            if platform and platform != 'local' and (not has_ip):
                ip = utils.get_ipv4_address()
                msg = f'nni_manager_ip is not set, please make sure {ip} is accessible from training machines'
                logging.getLogger('nni.experiment.config').warning(msg)

    def _validate_canonical(self):
        if False:
            while True:
                i = 10
        super()._validate_canonical()
        space_cnt = (self.search_space is not None) + (self.search_space_file is not None)
        if self.use_annotation and space_cnt != 0:
            raise ValueError('ExperimentConfig: search space must not be set when annotation is enabled')
        if not self.use_annotation and space_cnt < 1:
            raise ValueError('ExperimentConfig: search_space and search_space_file must be set one')
        assert self.trial_concurrency > 0
        assert self.max_experiment_duration is None or utils.parse_time(self.max_experiment_duration) > 0
        assert self.max_trial_number is None or self.max_trial_number > 0
        assert self.max_trial_duration is None or utils.parse_time(self.max_trial_duration) > 0
        assert self.log_level in ['fatal', 'error', 'warning', 'info', 'debug', 'trace']
        if type(self).__name__ != 'NasExperimentConfig':
            utils.validate_gpu_indices(self.tuner_gpu_indices)
            if self.tuner is None:
                raise ValueError('ExperimentConfig: tuner must be set')

def _load_search_space_file(search_space_path):
    if False:
        while True:
            i = 10
    content = Path(search_space_path).read_text(encoding='utf8')
    try:
        return json.loads(content)
    except Exception:
        return yaml.safe_load(content)