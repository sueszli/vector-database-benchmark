class MathFunction(object):

    def __init__(self, name, operator):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.operator = operator

    def __call__(self, *operands):
        if False:
            print('Hello World!')
        return self.operator(*operands)