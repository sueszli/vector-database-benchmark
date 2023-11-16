from __future__ import annotations
import json
from typing import Final
from localstack.services.stepfunctions.asl.component.eval_component import EvalComponent
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.comparison_operator_type import ComparisonOperatorType
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.operator.factory import OperatorFactory
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.operator.operator import Operator
from localstack.services.stepfunctions.asl.eval.environment import Environment

class ComparisonFunc(EvalComponent):

    def __init__(self, operator: ComparisonOperatorType, value: json):
        if False:
            i = 10
            return i + 15
        self.operator_type: Final[ComparisonOperatorType] = operator
        self.value: json = value

    def _eval_body(self, env: Environment) -> None:
        if False:
            for i in range(10):
                print('nop')
        value = self.value
        operator: Operator = OperatorFactory.get(self.operator_type)
        operator.eval(env=env, value=value)

    @staticmethod
    def _string_equals(env: Environment, value: json) -> None:
        if False:
            while True:
                i = 10
        val = env.stack.pop()
        res = str(val) == value
        env.stack.append(res)