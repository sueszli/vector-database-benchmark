from typing import Final
from localstack.services.stepfunctions.asl.component.eval_component import EvalComponent
from localstack.services.stepfunctions.asl.eval.environment import Environment
from localstack.services.stepfunctions.asl.utils.json_path import JSONPathUtils

class InputPath(EvalComponent):
    DEFAULT_PATH: Final[str] = '$'

    def __init__(self, input_path_src: str):
        if False:
            print('Hello World!')
        self.input_path_src: Final[str] = input_path_src

    def _eval_body(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        if self.input_path_src == InputPath.DEFAULT_PATH:
            value = env.inp
        else:
            value = JSONPathUtils.extract_json(self.input_path_src, env.inp)
        env.stack.append(value)