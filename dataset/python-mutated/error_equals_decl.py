from typing import Final
from localstack.services.stepfunctions.asl.component.common.error_name.error_name import ErrorName
from localstack.services.stepfunctions.asl.component.common.error_name.states_error_name import StatesErrorName
from localstack.services.stepfunctions.asl.component.common.error_name.states_error_name_type import StatesErrorNameType
from localstack.services.stepfunctions.asl.component.eval_component import EvalComponent
from localstack.services.stepfunctions.asl.eval.environment import Environment

class ErrorEqualsDecl(EvalComponent):
    """
    ErrorEquals value MUST be a non-empty array of Strings, which match Error Names.
    Each Retrier MUST contain a field named "ErrorEquals" whose value MUST be a non-empty array of Strings,
    which match Error Names.
    """
    _STATE_ALL_ERROR: Final[StatesErrorName] = StatesErrorName(typ=StatesErrorNameType.StatesALL)
    _STATE_TASK_ERROR: Final[StatesErrorName] = StatesErrorName(typ=StatesErrorNameType.StatesTaskFailed)

    def __init__(self, error_names: list[ErrorName]):
        if False:
            i = 10
            return i + 15
        if ErrorEqualsDecl._STATE_ALL_ERROR in error_names and len(error_names) > 1:
            raise ValueError(f"States.ALL must appear alone in the ErrorEquals array, got '{error_names}'.")
        self.error_names: list[ErrorName] = error_names

    def _eval_body(self, env: Environment) -> None:
        if False:
            while True:
                i = 10
        '\n        When a state reports an error, the interpreter scans through the Retriers and,\n        when the Error Name appears in the value of a Retrier’s "ErrorEquals" field, implements the retry policy\n        described in that Retrier.\n        This pops the error from the stack, and appends the bool of this check.\n        '
        error_name: ErrorName = env.stack.pop()
        if ErrorEqualsDecl._STATE_ALL_ERROR in self.error_names:
            res = True
        elif ErrorEqualsDecl._STATE_TASK_ERROR in self.error_names and (not isinstance(error_name, StatesErrorName)):
            res = True
        else:
            res = error_name in self.error_names
        env.stack.append(res)