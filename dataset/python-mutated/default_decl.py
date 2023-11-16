from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component

class DefaultDecl(Component):

    def __init__(self, state_name: str):
        if False:
            while True:
                i = 10
        self.state_name: Final[str] = state_name