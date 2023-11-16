from typing import Final
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.comparison import Comparison
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.comparison_func import ComparisonFunc
from localstack.services.stepfunctions.asl.component.state.state_choice.comparison.variable import Variable
from localstack.services.stepfunctions.asl.eval.environment import Environment

class ComparisonVariable(Comparison):
    variable: Final[Variable]
    comparison_function: Final[ComparisonFunc]

    def __init__(self, variable: Variable, func: ComparisonFunc):
        if False:
            for i in range(10):
                print('nop')
        self.variable = variable
        self.comparison_function = func

    def _eval_body(self, env: Environment) -> None:
        if False:
            for i in range(10):
                print('nop')
        variable: Variable = self.variable
        variable.eval(env)
        comparison_function: ComparisonFunc = self.comparison_function
        comparison_function.eval(env)