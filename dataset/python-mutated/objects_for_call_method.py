class MyObject:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.args = None

    def my_method(self, *args):
        if False:
            i = 10
            return i + 15
        if args == ('FAIL!',):
            raise RuntimeError('Expected failure')
        self.args = args

    def kwargs(self, arg1, arg2='default', **kwargs):
        if False:
            while True:
                i = 10
        kwargs = ['%s: %s' % item for item in sorted(kwargs.items())]
        return ', '.join([arg1, arg2] + kwargs)

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'String presentation of MyObject'
obj = MyObject()