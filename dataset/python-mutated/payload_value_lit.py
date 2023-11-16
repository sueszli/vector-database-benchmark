import abc
from typing import Any
from localstack.services.stepfunctions.asl.component.common.payload.payloadvalue.payload_value import PayloadValue
from localstack.services.stepfunctions.asl.eval.environment import Environment

class PayloadValueLit(PayloadValue, abc.ABC):
    val: Any

    def __init__(self, val: Any):
        if False:
            i = 10
            return i + 15
        self.val: Any = val

    def _eval_body(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        env.stack.append(self.val)