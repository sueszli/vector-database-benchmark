import datetime
import logging
from typing import Any, Final, Optional
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.comparison_operator_type import ComparisonOperatorType
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.operator.operator import Operator
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.variable import NoSuchVariable
from localstack.services.stepfunctions.asl.eval.environment import Environment
LOG = logging.getLogger(__name__)

class IsBoolean(Operator):

    @staticmethod
    def impl_name() -> str:
        if False:
            print('Hello World!')
        return str(ComparisonOperatorType.IsBoolean)

    @staticmethod
    def eval(env: Environment, value: Any) -> None:
        if False:
            print('Hello World!')
        variable = env.stack.pop()
        res = isinstance(variable, bool) is value
        env.stack.append(res)

class IsNull(Operator):

    @staticmethod
    def impl_name() -> str:
        if False:
            while True:
                i = 10
        return str(ComparisonOperatorType.IsNull)

    @staticmethod
    def eval(env: Environment, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        variable = env.stack.pop()
        is_null = variable is None and (not isinstance(variable, NoSuchVariable))
        res = is_null is value
        env.stack.append(res)

class IsNumeric(Operator):

    @staticmethod
    def impl_name() -> str:
        if False:
            while True:
                i = 10
        return str(ComparisonOperatorType.IsNumeric)

    @staticmethod
    def eval(env: Environment, value: Any) -> None:
        if False:
            return 10
        variable = env.stack.pop()
        res = (isinstance(variable, (int, float)) and (not isinstance(variable, bool))) is value
        env.stack.append(res)

class IsPresent(Operator):

    @staticmethod
    def impl_name() -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(ComparisonOperatorType.IsPresent)

    @staticmethod
    def eval(env: Environment, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        variable = env.stack.pop()
        res = isinstance(variable, NoSuchVariable) is not value
        env.stack.append(res)

class IsString(Operator):

    @staticmethod
    def impl_name() -> str:
        if False:
            print('Hello World!')
        return str(ComparisonOperatorType.IsString)

    @staticmethod
    def eval(env: Environment, value: Any) -> None:
        if False:
            return 10
        variable = env.stack.pop()
        res = isinstance(variable, str) is value
        env.stack.append(res)

class IsTimestamp(Operator):
    TIMESTAMP_FORMAT: Final[str] = '%Y-%m-%dT%H:%M:%SZ'

    @staticmethod
    def impl_name() -> str:
        if False:
            while True:
                i = 10
        return str(ComparisonOperatorType.IsTimestamp)

    @staticmethod
    def string_to_timestamp(string: str) -> Optional[datetime.datetime]:
        if False:
            print('Hello World!')
        try:
            return datetime.datetime.strptime(string, IsTimestamp.TIMESTAMP_FORMAT)
        except Exception:
            return None

    @staticmethod
    def is_timestamp(inp: Any) -> bool:
        if False:
            return 10
        return isinstance(inp, str) and IsTimestamp.string_to_timestamp(inp) is not None

    @staticmethod
    def eval(env: Environment, value: Any) -> None:
        if False:
            i = 10
            return i + 15
        variable = env.stack.pop()
        LOG.warning(f"State Choice's 'IsTimestamp' operator is not fully supported for input '{variable}' and target '{value}'.")
        res = IsTimestamp.is_timestamp(variable) is value
        env.stack.append(res)