from typing import Final
from localstack.services.stepfunctions.asl.component.state.state_wait.wait_function.wait_function import WaitFunction
from localstack.services.stepfunctions.asl.eval.environment import Environment

class Seconds(WaitFunction):

    def __init__(self, seconds: int):
        if False:
            return 10
        self.seconds: Final[int] = seconds

    def _get_wait_seconds(self, env: Environment) -> int:
        if False:
            print('Hello World!')
        return self.seconds