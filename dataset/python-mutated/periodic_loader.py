from typing import Mapping, TYPE_CHECKING
from collections import OrderedDict
from torch.utils.data import DataLoader
from catalyst.core.callback import Callback, CallbackOrder
if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

class PeriodicLoaderCallback(Callback):
    """Callback for runing loaders with specified period.
    To disable loader use ``0`` as period (if specified ``0`` for validation loader
    then will be raised an error).

    Args:
        kwargs: loader names and their run periods.

    For example, if you have ``train``, ``train_additional``,
    ``valid`` and ``valid_additional`` loaders and wan't to
    use ``train_additional`` every 2 epochs, ``valid`` - every
    3 epochs and ``valid_additional`` - every 5 epochs:

    .. code-block:: python

        from catalyst.dl import SupervisedRunner, PeriodicLoaderCallback
        runner = SupervisedRunner()
        runner.train(
            ...
            loaders={
                "train": ...,
                "train_additional": ...,
                "valid": ...,
                "valid_additional":...
            }
            ...
            callbacks=[
                ...
                PeriodicLoaderCallback(
                    train_additional=2,
                    valid=3,
                    valid_additional=5
                ),
                ...
            ]
            ...
        )

    """

    def __init__(self, valid_loader_key: str, valid_metric_key: str, minimize: bool, **kwargs):
        if False:
            i = 10
            return i + 15
        'Init.'
        super().__init__(order=CallbackOrder.internal)
        self.valid_loader_key: str = valid_loader_key
        self.valid_metric_key: str = valid_metric_key
        self.minimize_metric: bool = minimize
        self.loaders: Mapping[str, DataLoader] = OrderedDict()
        self.loader_periods = {}
        for (loader, period) in kwargs.items():
            if not isinstance(period, (int, float)):
                raise TypeError(f'Expected loader period type is int/float but got {type(period)}!')
            period = int(period)
            if period < 0:
                raise ValueError(f'Period should be >= 0, but got - {period}!')
            self.loader_periods[loader] = period

    def on_experiment_start(self, runner: 'IRunner') -> None:
        if False:
            for i in range(10):
                print('nop')
        'Collect information about loaders.\n\n        Args:\n            runner: current runner\n\n        Raises:\n            ValueError: if there are no loaders in epoch\n        '
        for (name, loader) in runner.loaders.items():
            self.loaders[name] = loader
        is_loaders_match = all((loader in runner.loaders for loader in self.loader_periods.keys()))
        is_same_loaders_number = len(self.loader_periods) == len(runner.loaders)
        if is_same_loaders_number and is_loaders_match:
            zero_loaders_epochs = list(filter(lambda n: all((p == 0 or n % p != 0 for p in self.loader_periods.values())), range(1, runner.num_epochs + 1)))
            if len(zero_loaders_epochs) > 0:
                epoch_with_err = zero_loaders_epochs[0]
                raise ValueError(f'There will be no loaders in epoch {epoch_with_err}!')
        if self.loader_periods.get(self.valid_loader_key, 1) < 1:
            raise ValueError(f"Period for a validation loader ('{self.valid_loader_key}') should be > 0!")

    def on_epoch_start(self, runner: 'IRunner') -> None:
        if False:
            return 10
        '\n        Set loaders for current epoch.\n        If validation is not required then the first loader\n        from loaders used in current epoch will be used\n        as validation loader.\n        Metrics from the latest epoch with true\n        validation loader will be used\n        in the epochs where this loader is missing.\n\n        Args:\n            runner: current runner\n\n        Raises:\n            ValueError: if there are no loaders in epoch\n        '
        epoch_step = runner.epoch_step
        epoch_loaders = OrderedDict()
        for (name, loader) in self.loaders.items():
            period = self.loader_periods.get(name, 1)
            if period > 0 and epoch_step % period == 0:
                epoch_loaders[name] = loader
        if len(epoch_loaders) == 0:
            raise ValueError(f'There is no loaders in epoch {epoch_step}!')
        runner.loaders = epoch_loaders

    def on_epoch_end(self, runner: 'IRunner') -> None:
        if False:
            print('Hello World!')
        'Check if validation metric should be dropped for current epoch.\n\n        Args:\n            runner: current runner\n        '
        if self.valid_loader_key not in runner.loaders:
            runner.epoch_metrics[self.valid_loader_key] = {self.valid_metric_key: float('+inf') if self.minimize_metric else float('-inf')}
__all__ = ['PeriodicLoaderCallback']