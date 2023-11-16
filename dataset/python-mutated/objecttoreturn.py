class ObjectToReturn:

    def __init__(self, name):
        if False:
            return 10
        self.name = name

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

    def exception(self, name, msg=''):
        if False:
            return 10
        try:
            exception = getattr(__builtins__, name)
        except AttributeError:
            exception = __builtins__[name]
        raise exception(msg)