class SpeedupError(Exception):

    def __init__(self, msg):
        if False:
            return 10
        self.msg = msg

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.msg)

class EmptyLayerError(SpeedupError):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(EmptyLayerError, self).__init__('Pruning a Layer to empty is not legal')

class ShapeMisMatchError(SpeedupError):

    def __init__(self):
        if False:
            print('Hello World!')
        super(ShapeMisMatchError, self).__init__('Shape mismatch!')

class InputsNumberError(SpeedupError):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(InputsNumberError, self).__init__('The number of the inputs of the target OP is wrong')

class OutputTypeError(SpeedupError):

    def __init__(self, current_type, target_type):
        if False:
            print('Hello World!')
        msg = f'The output type should be {str(target_type)}, but {str(current_type)} founded'
        super(OutputTypeError, self).__init__(msg)

class UnBalancedGroupError(SpeedupError):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'The number remained filters in each group is different'
        super(UnBalancedGroupError, self).__init__(msg)