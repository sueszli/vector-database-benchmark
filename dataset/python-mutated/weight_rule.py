from __future__ import annotations
from enum import Enum
from airflow.compat.functools import cache

class WeightRule(str, Enum):
    """Weight rules."""
    DOWNSTREAM = 'downstream'
    UPSTREAM = 'upstream'
    ABSOLUTE = 'absolute'

    @classmethod
    def is_valid(cls, weight_rule: str) -> bool:
        if False:
            print('Hello World!')
        'Check if weight rule is valid.'
        return weight_rule in cls.all_weight_rules()

    @classmethod
    @cache
    def all_weight_rules(cls) -> set[str]:
        if False:
            i = 10
            return i + 15
        'Return all weight rules.'
        return set(cls.__members__.values())

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.value