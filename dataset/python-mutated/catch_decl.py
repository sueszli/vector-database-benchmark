from typing import Final
from localstack.services.stepfunctions.asl.component.common.catch.catch_outcome import CatchOutcome
from localstack.services.stepfunctions.asl.component.common.catch.catcher_decl import CatcherDecl
from localstack.services.stepfunctions.asl.component.common.catch.catcher_outcome import CatcherOutcome, CatcherOutcomeCaught
from localstack.services.stepfunctions.asl.component.eval_component import EvalComponent
from localstack.services.stepfunctions.asl.eval.environment import Environment

class CatchDecl(EvalComponent):

    def __init__(self, catchers: list[CatcherDecl]):
        if False:
            for i in range(10):
                print('nop')
        self.catchers: Final[list[CatcherDecl]] = catchers

    def _eval_body(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        for catcher in self.catchers:
            catcher.eval(env)
            catcher_outcome: CatcherOutcome = env.stack.pop()
            if isinstance(catcher_outcome, CatcherOutcomeCaught):
                env.stack.append(CatchOutcome.Caught)
                return
        env.stack.append(CatchOutcome.NotCaught)