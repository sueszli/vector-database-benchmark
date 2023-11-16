class InitWithOnlyNamedOnlyArgs:

    def __init__(self, *, a=1, b):
        if False:
            print('Hello World!')
        'xxx'

    def kw(self):
        if False:
            i = 10
            return i + 15
        pass