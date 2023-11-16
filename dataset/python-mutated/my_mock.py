class MyMock:

    def __init__(self, methods=None):
        if False:
            print('Hello World!')
        self.methods = methods or {}

    def __getattr__(self, item):
        if False:
            return 10

        def not_implemented():
            if False:
                i = 10
                return i + 15
            raise NotImplementedError
        if item in self.methods:
            return self.methods[item]
        return not_implemented