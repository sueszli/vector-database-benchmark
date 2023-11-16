"""The factory module provides a factory class for constructing concrete issue managers
and a decorator for registering new issue managers.

This module provides the :py:meth:`register` decorator for users to register new subclasses of
:py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`
in the registry. Each IssueManager detects some particular type of issue in a dataset.


Note
----

The :class:`REGISTRY` variable is used by the factory class to keep track
of registered issue managers.
The factory class is used as an implementation detail by
:py:class:`Datalab <cleanlab.datalab.datalab.Datalab>`,
which provides a simplified API for constructing concrete issue managers.
:py:class:`Datalab <cleanlab.datalab.datalab.Datalab>` is intended to be used by users
and provides detailed documentation on how to use the API.

Warning
-------
Neither the :class:`REGISTRY` variable nor the factory class should be used directly by users.
"""
from __future__ import annotations
from typing import Dict, List, Type
from cleanlab.datalab.internal.issue_manager import IssueManager, LabelIssueManager, NearDuplicateIssueManager, OutlierIssueManager, NonIIDIssueManager, ClassImbalanceIssueManager, DataValuationIssueManager
REGISTRY: Dict[str, Type[IssueManager]] = {'outlier': OutlierIssueManager, 'label': LabelIssueManager, 'near_duplicate': NearDuplicateIssueManager, 'non_iid': NonIIDIssueManager, 'class_imbalance': ClassImbalanceIssueManager, 'data_valuation': DataValuationIssueManager}
'Registry of issue managers that can be constructed from a string\nand used in the Datalab class.\n\n:meta hide-value:\n\nCurrently, the following issue managers are registered by default:\n\n- ``"outlier"``: :py:class:`OutlierIssueManager <cleanlab.datalab.internal.issue_manager.outlier.OutlierIssueManager>`\n- ``"label"``: :py:class:`LabelIssueManager <cleanlab.datalab.internal.issue_manager.label.LabelIssueManager>`\n- ``"near_duplicate"``: :py:class:`NearDuplicateIssueManager <cleanlab.datalab.internal.issue_manager.duplicate.NearDuplicateIssueManager>`\n- ``"non_iid"``: :py:class:`NonIIDIssueManager <cleanlab.datalab.internal.issue_manager.noniid.NonIIDIssueManager>`\n\nWarning\n-------\nThis variable should not be used directly by users.\n'

class _IssueManagerFactory:
    """Factory class for constructing concrete issue managers."""

    @classmethod
    def from_str(cls, issue_type: str) -> Type[IssueManager]:
        if False:
            print('Hello World!')
        'Constructs a concrete issue manager class from a string.'
        if isinstance(issue_type, list):
            raise ValueError('issue_type must be a string, not a list. Try using from_list instead.')
        if issue_type not in REGISTRY:
            raise ValueError(f'Invalid issue type: {issue_type}')
        return REGISTRY[issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str]) -> List[Type[IssueManager]]:
        if False:
            while True:
                i = 10
        'Constructs a list of concrete issue manager classes from a list of strings.'
        return [cls.from_str(issue_type) for issue_type in issue_types]

def register(cls: Type[IssueManager]) -> Type[IssueManager]:
    if False:
        for i in range(10):
            print('nop')
    'Registers the issue manager factory.\n\n    Parameters\n    ----------\n    cls :\n        A subclass of\n        :py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`.\n\n    Returns\n    -------\n    cls :\n        The same class that was passed in.\n\n    Example\n    -------\n\n    When defining a new subclass of\n    :py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`,\n    you can register it like so:\n\n    .. code-block:: python\n\n        from cleanlab import IssueManager\n        from cleanlab.datalab.internal.issue_manager_factory import register\n\n        @register\n        class MyIssueManager(IssueManager):\n            issue_name: str = "my_issue"\n            def find_issues(self, **kwargs):\n                # Some logic to find issues\n                pass\n\n    or in a function call:\n\n    .. code-block:: python\n\n        from cleanlab import IssueManager\n        from cleanlab.datalab.internal.issue_manager_factory import register\n\n        class MyIssueManager(IssueManager):\n            issue_name: str = "my_issue"\n            def find_issues(self, **kwargs):\n                # Some logic to find issues\n                pass\n\n        register(MyIssueManager)\n    '
    name: str = str(cls.issue_name)
    if name in REGISTRY:
        print(f'Warning: Overwriting existing issue manager {name} with {cls}. This may cause unexpected behavior.')
    if not issubclass(cls, IssueManager):
        raise ValueError(f'Class {cls} must be a subclass of IssueManager')
    REGISTRY[name] = cls
    return cls