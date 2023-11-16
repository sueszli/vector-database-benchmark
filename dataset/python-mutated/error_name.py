from __future__ import annotations
import abc
from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component

class ErrorName(Component, abc.ABC):

    def __init__(self, error_name: str):
        if False:
            return 10
        self.error_name: Final[str] = error_name

    def matches(self, error_name: str) -> bool:
        if False:
            i = 10
            return i + 15
        return self.error_name == error_name

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ErrorName):
            return self.matches(other.error_name)
        return False