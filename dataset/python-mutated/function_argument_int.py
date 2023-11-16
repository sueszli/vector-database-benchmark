from localstack.services.stepfunctions.asl.component.intrinsic.argument.function_argument import FunctionArgument

class FunctionArgumentInt(FunctionArgument):
    _value: int

    def __init__(self, integer: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(value=integer)