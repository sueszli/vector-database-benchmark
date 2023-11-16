from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component

class ErrorDecl(Component):

    def __init__(self, error: str):
        if False:
            i = 10
            return i + 15
        self.error: Final[str] = error