from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component

class CauseDecl(Component):

    def __init__(self, cause: str):
        if False:
            while True:
                i = 10
        self.cause: Final[str] = cause