"""Registry for issues."""
from __future__ import annotations
import importlib
import os
from core import feconf
from core.platform import models
from extensions.issues import base
from typing import Dict, List
MYPY = False
if MYPY:
    from mypy_imports import stats_models
(stats_models,) = models.Registry.import_models([models.Names.STATISTICS])

class Registry:
    """Registry of all issues."""
    _issues: Dict[str, base.BaseExplorationIssueSpec] = {}

    @classmethod
    def get_all_issue_types(cls) -> List[str]:
        if False:
            print('Hello World!')
        'Get a list of all issue types.\n\n        Returns:\n            list(str). The list of all allowed issue types.\n        '
        return stats_models.ALLOWED_ISSUE_TYPES

    @classmethod
    def _refresh(cls) -> None:
        if False:
            print('Hello World!')
        'Initializes the mapping between issue types to instances of the issue\n        classes.\n        '
        cls._issues.clear()
        for issue_type in cls.get_all_issue_types():
            module_path_parts = feconf.ISSUES_DIR.split(os.sep)
            module_path_parts.extend([issue_type, issue_type])
            module = importlib.import_module('.'.join(module_path_parts))
            clazz = getattr(module, issue_type)
            ancestor_names = [base_class.__name__ for base_class in clazz.__bases__]
            if 'BaseExplorationIssueSpec' in ancestor_names:
                cls._issues[clazz.__name__] = clazz()

    @classmethod
    def get_all_issues(cls) -> List[base.BaseExplorationIssueSpec]:
        if False:
            return 10
        'Get a list of instances of all issues.\n\n        Returns:\n            list(*). A list of all issue class instances. Classes all have\n            "BaseExplorationIssueSpec" as an ancestor class.\n        '
        if len(cls._issues) == 0:
            cls._refresh()
        return list(cls._issues.values())

    @classmethod
    def get_issue_by_type(cls, issue_type: str) -> base.BaseExplorationIssueSpec:
        if False:
            print('Hello World!')
        'Gets an issue by its type.\n\n        Refreshes once if the issue is not found; subsequently, throws a\n        KeyError.\n\n        Args:\n            issue_type: str. Type of the issue.\n\n        Returns:\n            *. An instance of the corresponding issue class. This class has\n            "BaseExplorationIssueSpec" as an ancestor class.\n        '
        if issue_type not in cls._issues:
            cls._refresh()
        return cls._issues[issue_type]