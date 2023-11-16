from typing import List
from ray.data._internal.logical.interfaces.optimizer import Rule

def add_user_provided_logical_rules(default_rules: List[Rule]) -> List[Rule]:
    if False:
        i = 10
        return i + 15
    '\n    Users can provide extra logical optimization rules here\n    to be used in `LogicalOptimizer`.\n\n    Args:\n        default_rules: the default logical optimization rules.\n\n    Returns:\n        The final logical optimization rules to be used in `LogicalOptimizer`.\n    '
    return default_rules

def add_user_provided_physical_rules(default_rules: List[Rule]) -> List[Rule]:
    if False:
        while True:
            i = 10
    '\n    Users can provide extra physical optimization rules here\n    to be used in `PhysicalOptimizer`.\n\n    Args:\n        default_rules: the default physical optimization rules.\n\n    Returns:\n        The final physical optimization rules to be used in `PhysicalOptimizer`.\n    '
    return default_rules