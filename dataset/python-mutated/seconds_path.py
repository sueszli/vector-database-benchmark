from typing import Final
from jsonpath_ng import parse
from localstack.services.stepfunctions.asl.component.state.state_wait.wait_function.wait_function import WaitFunction
from localstack.services.stepfunctions.asl.eval.environment import Environment

class SecondsPath(WaitFunction):

    def __init__(self, path: str):
        if False:
            for i in range(10):
                print('nop')
        self.path: Final[str] = path

    def _get_wait_seconds(self, env: Environment) -> int:
        if False:
            i = 10
            return i + 15
        input_expr = parse(self.path)
        seconds = input_expr.find(env.inp)
        return seconds