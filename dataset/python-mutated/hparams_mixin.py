import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union
from lightning.pytorch.utilities.parsing import AttributeDict, save_hyperparameters
_PRIMITIVE_TYPES = (bool, int, float, str)
_ALLOWED_CONFIG_TYPES = (AttributeDict, MutableMapping, Namespace)

class HyperparametersMixin:
    __jit_unused_properties__: List[str] = ['hparams', 'hparams_initial']

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self._log_hyperparams = False

    def save_hyperparameters(self, *args: Any, ignore: Optional[Union[Sequence[str], str]]=None, frame: Optional[types.FrameType]=None, logger: bool=True) -> None:
        if False:
            print('Hello World!')
        'Save arguments to ``hparams`` attribute.\n\n        Args:\n            args: single object of `dict`, `NameSpace` or `OmegaConf`\n                or string names or arguments from class ``__init__``\n            ignore: an argument name or a list of argument names from\n                class ``__init__`` to be ignored\n            frame: a frame object. Default is None\n            logger: Whether to send the hyperparameters to the logger. Default: True\n\n        Example::\n            >>> from lightning.pytorch.core.mixins import HyperparametersMixin\n            >>> class ManuallyArgsModel(HyperparametersMixin):\n            ...     def __init__(self, arg1, arg2, arg3):\n            ...         super().__init__()\n            ...         # manually assign arguments\n            ...         self.save_hyperparameters(\'arg1\', \'arg3\')\n            ...     def forward(self, *args, **kwargs):\n            ...         ...\n            >>> model = ManuallyArgsModel(1, \'abc\', 3.14)\n            >>> model.hparams\n            "arg1": 1\n            "arg3": 3.14\n\n            >>> from lightning.pytorch.core.mixins import HyperparametersMixin\n            >>> class AutomaticArgsModel(HyperparametersMixin):\n            ...     def __init__(self, arg1, arg2, arg3):\n            ...         super().__init__()\n            ...         # equivalent automatic\n            ...         self.save_hyperparameters()\n            ...     def forward(self, *args, **kwargs):\n            ...         ...\n            >>> model = AutomaticArgsModel(1, \'abc\', 3.14)\n            >>> model.hparams\n            "arg1": 1\n            "arg2": abc\n            "arg3": 3.14\n\n            >>> from lightning.pytorch.core.mixins import HyperparametersMixin\n            >>> class SingleArgModel(HyperparametersMixin):\n            ...     def __init__(self, params):\n            ...         super().__init__()\n            ...         # manually assign single argument\n            ...         self.save_hyperparameters(params)\n            ...     def forward(self, *args, **kwargs):\n            ...         ...\n            >>> model = SingleArgModel(Namespace(p1=1, p2=\'abc\', p3=3.14))\n            >>> model.hparams\n            "p1": 1\n            "p2": abc\n            "p3": 3.14\n\n            >>> from lightning.pytorch.core.mixins import HyperparametersMixin\n            >>> class ManuallyArgsModel(HyperparametersMixin):\n            ...     def __init__(self, arg1, arg2, arg3):\n            ...         super().__init__()\n            ...         # pass argument(s) to ignore as a string or in a list\n            ...         self.save_hyperparameters(ignore=\'arg2\')\n            ...     def forward(self, *args, **kwargs):\n            ...         ...\n            >>> model = ManuallyArgsModel(1, \'abc\', 3.14)\n            >>> model.hparams\n            "arg1": 1\n            "arg3": 3.14\n\n        '
        self._log_hyperparams = logger
        if not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back
        save_hyperparameters(self, *args, ignore=ignore, frame=frame)

    def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
        if False:
            i = 10
            return i + 15
        hp = self._to_hparams_dict(hp)
        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    @staticmethod
    def _to_hparams_dict(hp: Union[MutableMapping, Namespace, str]) -> Union[MutableMapping, AttributeDict]:
        if False:
            return 10
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, _PRIMITIVE_TYPES):
            raise ValueError(f'Primitives {_PRIMITIVE_TYPES} are not allowed.')
        elif not isinstance(hp, _ALLOWED_CONFIG_TYPES):
            raise ValueError(f'Unsupported config type of {type(hp)}.')
        return hp

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        if False:
            for i in range(10):
                print('nop')
        'The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For\n        the frozen set of initial hyperparameters, use :attr:`hparams_initial`.\n\n        Returns:\n            Mutable hyperparameters dictionary\n\n        '
        if not hasattr(self, '_hparams'):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        if False:
            print('Hello World!')
        'The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.\n        Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.\n\n        Returns:\n            AttributeDict: immutable initial hyperparameters\n\n        '
        if not hasattr(self, '_hparams_initial'):
            return AttributeDict()
        return copy.deepcopy(self._hparams_initial)