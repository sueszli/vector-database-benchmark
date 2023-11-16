"""Registry for actions."""
from __future__ import annotations
import importlib
import os
from core import feconf
from core.platform import models
from extensions.actions import base
from typing import Dict, List
MYPY = False
if MYPY:
    from mypy_imports import stats_models
(stats_models,) = models.Registry.import_models([models.Names.STATISTICS])

class Registry:
    """Registry of all actions."""
    _actions: Dict[str, base.BaseLearnerActionSpec] = {}

    @classmethod
    def get_all_action_types(cls) -> List[str]:
        if False:
            return 10
        'Get a list of all action types.\n\n        Returns:\n            list(str). The list of all allowed action types.\n        '
        return stats_models.ALLOWED_ACTION_TYPES

    @classmethod
    def _refresh(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes the mapping between action types to instances of the\n        action classes.\n        '
        cls._actions.clear()
        for action_type in cls.get_all_action_types():
            module_path_parts = feconf.ACTIONS_DIR.split(os.sep)
            module_path_parts.extend([action_type, action_type])
            module = importlib.import_module('.'.join(module_path_parts))
            clazz = getattr(module, action_type)
            ancestor_names = [base_class.__name__ for base_class in clazz.__bases__]
            if 'BaseLearnerActionSpec' in ancestor_names:
                cls._actions[clazz.__name__] = clazz()

    @classmethod
    def get_all_actions(cls) -> List[base.BaseLearnerActionSpec]:
        if False:
            return 10
        'Get a list of instances of all actions.\n\n        Returns:\n            list(*). A list of all action class instances. Classes all have\n            "BaseLearnerActionSpec" as an ancestor class.\n        '
        if len(cls._actions) == 0:
            cls._refresh()
        return list(cls._actions.values())

    @classmethod
    def get_action_by_type(cls, action_type: str) -> base.BaseLearnerActionSpec:
        if False:
            while True:
                i = 10
        'Gets an action by its type.\n\n        Refreshes once if the action is not found; subsequently, throws a\n        KeyError.\n\n        Args:\n            action_type: str. Type of the action.\n\n        Returns:\n            *. An instance of the corresponding action class. This class has\n            "BaseLearnerActionSpec" as an ancestor class.\n        '
        if action_type not in cls._actions:
            cls._refresh()
        return cls._actions[action_type]