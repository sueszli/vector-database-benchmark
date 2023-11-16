import abc
from typing import Any, Final
from localstack.services.stepfunctions.asl.component.common.payload.payloadvalue.payload_value import PayloadValue
from localstack.services.stepfunctions.asl.eval.environment import Environment

class PayloadBinding(PayloadValue, abc.ABC):

    def __init__(self, field: str):
        if False:
            for i in range(10):
                print('nop')
        self.field: Final[str] = field

    @abc.abstractmethod
    def _eval_val(self, env: Environment) -> Any:
        if False:
            i = 10
            return i + 15
        ...

    def _eval_body(self, env: Environment) -> None:
        if False:
            return 10
        val = self._eval_val(env=env)
        cnt: dict = env.stack.pop()
        cnt[self.field] = val
        env.stack.append(cnt)