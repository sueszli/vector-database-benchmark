import base64
from typing import Final
from localstack.services.stepfunctions.asl.component.intrinsic.argument.function_argument_list import FunctionArgumentList
from localstack.services.stepfunctions.asl.component.intrinsic.function.statesfunction.states_function import StatesFunction
from localstack.services.stepfunctions.asl.component.intrinsic.functionname.state_fuinction_name_types import StatesFunctionNameType
from localstack.services.stepfunctions.asl.component.intrinsic.functionname.states_function_name import StatesFunctionName
from localstack.services.stepfunctions.asl.eval.environment import Environment

class Base64Decode(StatesFunction):
    MAX_INPUT_CHAR_LEN: Final[int] = 10000

    def __init__(self, arg_list: FunctionArgumentList):
        if False:
            return 10
        super().__init__(states_name=StatesFunctionName(function_type=StatesFunctionNameType.Base64Decode), arg_list=arg_list)
        if arg_list.size != 1:
            raise ValueError(f"Expected 1 argument for function type '{type(self)}', but got: '{arg_list}'.")

    def _eval_body(self, env: Environment) -> None:
        if False:
            return 10
        self.arg_list.eval(env=env)
        base64_string: str = env.stack.pop()
        if len(base64_string) > self.MAX_INPUT_CHAR_LEN:
            raise ValueError(f"Maximum input string for function type '{type(self)}' is '{self.MAX_INPUT_CHAR_LEN}', but got '{len(base64_string)}'.")
        base64_string_bytes = base64_string.encode('ascii')
        string_bytes = base64.b64decode(base64_string_bytes)
        string = string_bytes.decode('ascii')
        env.stack.append(string)