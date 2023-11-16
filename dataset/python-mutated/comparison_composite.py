from __future__ import annotations
import abc
from enum import Enum
from typing import Any, Final
from localstack.services.stepfunctions.asl.antlr.runtime.ASLLexer import ASLLexer
from localstack.services.stepfunctions.asl.component.state.state_choice.choice_rule import ChoiceRule
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.comparison import Comparison
from localstack.services.stepfunctions.asl.eval.environment import Environment
from localstack.services.stepfunctions.asl.parse.typed_props import TypedProps

class ComparisonCompositeProps(TypedProps):

    def add(self, instance: Any) -> None:
        if False:
            i = 10
            return i + 15
        inst_type = type(instance)
        if issubclass(inst_type, ComparisonComposite):
            super()._add(ComparisonComposite, instance)
            return
        super().add(instance)

class ComparisonComposite(Comparison, abc.ABC):

    class ChoiceOp(Enum):
        And = ASLLexer.AND
        Or = ASLLexer.OR
        Not = ASLLexer.NOT
    operator: Final[ComparisonComposite.ChoiceOp]

    def __init__(self, operator: ComparisonComposite.ChoiceOp):
        if False:
            print('Hello World!')
        self.operator = operator

class ComparisonCompositeSingle(ComparisonComposite, abc.ABC):
    rule: Final[ChoiceRule]

    def __init__(self, operator: ComparisonComposite.ChoiceOp, rule: ChoiceRule):
        if False:
            while True:
                i = 10
        super(ComparisonCompositeSingle, self).__init__(operator=operator)
        self.rule = rule

class ComparisonCompositeMulti(ComparisonComposite, abc.ABC):
    rules: Final[list[ChoiceRule]]

    def __init__(self, operator: ComparisonComposite.ChoiceOp, rules: list[ChoiceRule]):
        if False:
            for i in range(10):
                print('nop')
        super(ComparisonCompositeMulti, self).__init__(operator=operator)
        self.rules = rules

class ComparisonCompositeNot(ComparisonCompositeSingle):

    def __init__(self, rule: ChoiceRule):
        if False:
            while True:
                i = 10
        super(ComparisonCompositeNot, self).__init__(operator=ComparisonComposite.ChoiceOp.Not, rule=rule)

    def _eval_body(self, env: Environment) -> None:
        if False:
            print('Hello World!')
        self.rule.eval(env)
        tmp: bool = env.stack.pop()
        res = tmp is False
        env.stack.append(res)

class ComparisonCompositeAnd(ComparisonCompositeMulti):

    def __init__(self, rules: list[ChoiceRule]):
        if False:
            while True:
                i = 10
        super(ComparisonCompositeAnd, self).__init__(operator=ComparisonComposite.ChoiceOp.And, rules=rules)

    def _eval_body(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        res = True
        for rule in self.rules:
            rule.eval(env)
            rule_out = env.stack.pop()
            if not rule_out:
                res = False
                break
        env.stack.append(res)

class ComparisonCompositeOr(ComparisonCompositeMulti):

    def __init__(self, rules: list[ChoiceRule]):
        if False:
            i = 10
            return i + 15
        super(ComparisonCompositeOr, self).__init__(operator=ComparisonComposite.ChoiceOp.Or, rules=rules)

    def _eval_body(self, env: Environment) -> None:
        if False:
            print('Hello World!')
        res = False
        for rule in self.rules:
            rule.eval(env)
            rule_out = env.stack.pop()
            res = res or rule_out
            if res:
                break
        env.stack.append(res)