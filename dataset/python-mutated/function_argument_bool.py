from localstack.services.stepfunctions.asl.component.intrinsic.argument.function_argument import FunctionArgument

class FunctionArgumentBool(FunctionArgument):
    _value: bool

    def __init__(self, boolean: bool):
        if False:
            return 10
        super().__init__(value=boolean)