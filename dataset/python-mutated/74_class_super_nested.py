def fun():
    if False:
        i = 10
        return i + 15

    class Foo:

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()

        def no_super(self):
            if False:
                return 10
            return