from typing import Final
from localstack.services.stepfunctions.asl.component.component import Component
from localstack.services.stepfunctions.asl.component.state.state_choice.choice_rule import ChoiceRule

class ChoicesDecl(Component):

    def __init__(self, rules: list[ChoiceRule]):
        if False:
            for i in range(10):
                print('nop')
        self.rules: Final[list[ChoiceRule]] = rules