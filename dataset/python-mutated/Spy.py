import logging

class Spy:

    def __init__(self, obj, func_name):
        if False:
            print('Hello World!')
        self.obj = obj
        self.__name__ = func_name
        self.func_original = getattr(self.obj, func_name)
        self.calls = []

    def __enter__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        logging.debug('Spy started')

        def loggedFunc(cls, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            call = dict(enumerate(args, 1))
            call[0] = cls
            call.update(kwargs)
            logging.debug('Spy call: %s' % call)
            self.calls.append(call)
            return self.func_original(cls, *args, **kwargs)
        setattr(self.obj, self.__name__, loggedFunc)
        return self.calls

    def __exit__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        setattr(self.obj, self.__name__, self.func_original)