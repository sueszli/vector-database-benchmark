from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component

class CSVHeaders(Component):
    header_names: Final[list[str]]

    def __init__(self, header_names: list[str]):
        if False:
            for i in range(10):
                print('nop')
        self.header_names = header_names