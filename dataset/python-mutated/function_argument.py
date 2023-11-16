import abc
from typing import Any, Optional
from localstack.services.stepfunctions.asl.component.eval_component import EvalComponent
from localstack.services.stepfunctions.asl.eval.environment import Environment

class FunctionArgument(EvalComponent, abc.ABC):
    _value: Optional[Any]

    def __init__(self, value: Any=None):
        if False:
            while True:
                i = 10
        self._value = value

    def _eval_body(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        env.stack.append(self._value)