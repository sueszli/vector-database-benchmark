import copy
from typing import Any
from localstack.services.stepfunctions.asl.component.intrinsic.argument.function_argument_list import FunctionArgumentList
from localstack.services.stepfunctions.asl.component.intrinsic.function.statesfunction.states_function import StatesFunction
from localstack.services.stepfunctions.asl.component.intrinsic.functionname.state_function_name_types import StatesFunctionNameType
from localstack.services.stepfunctions.asl.component.intrinsic.functionname.states_function_name import StatesFunctionName
from localstack.services.stepfunctions.asl.eval.environment import Environment

class JsonMerge(StatesFunction):

    def __init__(self, arg_list: FunctionArgumentList):
        if False:
            print('Hello World!')
        super().__init__(states_name=StatesFunctionName(function_type=StatesFunctionNameType.JsonMerge), arg_list=arg_list)
        if arg_list.size != 3:
            raise ValueError(f"Expected 3 arguments for function type '{type(self)}', but got: '{arg_list}'.")

    @staticmethod
    def _validate_is_deep_merge_argument(is_deep_merge: Any) -> None:
        if False:
            while True:
                i = 10
        if not isinstance(is_deep_merge, bool):
            raise TypeError(f"Expected boolean value for deep merge mode, but got: '{is_deep_merge}'.")
        if is_deep_merge:
            raise NotImplementedError('Currently, Step Functions only supports the shallow merging mode; therefore, you must specify the boolean value as false.')

    @staticmethod
    def _validate_merge_argument(argument: Any, num: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(argument, dict):
            raise TypeError(f"Expected a JSON object the argument {num}, but got: '{argument}'.")

    def _eval_body(self, env: Environment) -> None:
        if False:
            while True:
                i = 10
        self.arg_list.eval(env=env)
        is_deep_merge = env.stack.pop()
        self._validate_is_deep_merge_argument(is_deep_merge)
        snd = env.stack.pop()
        self._validate_merge_argument(snd, 2)
        fst = env.stack.pop()
        self._validate_merge_argument(snd, 2)
        merged = copy.deepcopy(fst)
        merged.update(snd)
        env.stack.append(merged)