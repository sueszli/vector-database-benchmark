from robot.errors import DataError

class Resolvable:

    def resolve(self, variables):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def report_error(self, error):
        if False:
            print('Hello World!')
        raise DataError(error)

class GlobalVariableValue(Resolvable):

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def resolve(self, variables):
        if False:
            return 10
        return self.value