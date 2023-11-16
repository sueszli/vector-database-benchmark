class Foo:

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def no_super(self):
        if False:
            print('Hello World!')
        return