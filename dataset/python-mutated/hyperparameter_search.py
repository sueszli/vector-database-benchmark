from .integrations import is_optuna_available, is_ray_available, is_sigopt_available, is_wandb_available, run_hp_search_optuna, run_hp_search_ray, run_hp_search_sigopt, run_hp_search_wandb
from .trainer_utils import HPSearchBackend, default_hp_space_optuna, default_hp_space_ray, default_hp_space_sigopt, default_hp_space_wandb
from .utils import logging
logger = logging.get_logger(__name__)

class HyperParamSearchBackendBase:
    name: str
    pip_package: str = None

    @staticmethod
    def is_available():
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def default_hp_space(self, trial):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def ensure_available(self):
        if False:
            print('Hello World!')
        if not self.is_available():
            raise RuntimeError(f'You picked the {self.name} backend, but it is not installed. Run {self.pip_install()}.')

    @classmethod
    def pip_install(cls):
        if False:
            for i in range(10):
                print('nop')
        return f'`pip install {cls.pip_package or cls.name}`'

class OptunaBackend(HyperParamSearchBackendBase):
    name = 'optuna'

    @staticmethod
    def is_available():
        if False:
            print('Hello World!')
        return is_optuna_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return run_hp_search_optuna(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        if False:
            return 10
        return default_hp_space_optuna(trial)

class RayTuneBackend(HyperParamSearchBackendBase):
    name = 'ray'
    pip_package = "'ray[tune]'"

    @staticmethod
    def is_available():
        if False:
            return 10
        return is_ray_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        if False:
            print('Hello World!')
        return run_hp_search_ray(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        if False:
            return 10
        return default_hp_space_ray(trial)

class SigOptBackend(HyperParamSearchBackendBase):
    name = 'sigopt'

    @staticmethod
    def is_available():
        if False:
            i = 10
            return i + 15
        return is_sigopt_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        if False:
            print('Hello World!')
        return run_hp_search_sigopt(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        if False:
            while True:
                i = 10
        return default_hp_space_sigopt(trial)

class WandbBackend(HyperParamSearchBackendBase):
    name = 'wandb'

    @staticmethod
    def is_available():
        if False:
            return 10
        return is_wandb_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        if False:
            i = 10
            return i + 15
        return default_hp_space_wandb(trial)
ALL_HYPERPARAMETER_SEARCH_BACKENDS = {HPSearchBackend(backend.name): backend for backend in [OptunaBackend, RayTuneBackend, SigOptBackend, WandbBackend]}

def default_hp_search_backend() -> str:
    if False:
        i = 10
        return i + 15
    available_backends = [backend for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values() if backend.is_available()]
    if len(available_backends) > 0:
        name = available_backends[0].name
        if len(available_backends) > 1:
            logger.info(f'{len(available_backends)} hyperparameter search backends available. Using {name} as the default.')
        return name
    raise RuntimeError('No hyperparameter search backend available.\n' + '\n'.join((f' - To install {backend.name} run {backend.pip_install()}' for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values())))