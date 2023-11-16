from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component

class MaxConcurrency(Component):
    DEFAULT: Final[int] = 0

    def __init__(self, num: int=DEFAULT):
        if False:
            for i in range(10):
                print('nop')
        self.num: Final[int] = num