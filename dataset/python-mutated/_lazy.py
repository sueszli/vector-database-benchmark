import functools

class lazy_property(object):

    def __init__(self, func):
        if False:
            print('Hello World!')
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        if False:
            while True:
                i = 10
        if instance is None:
            return self
        value = self.func(instance)
        setattr(instance, self.func.__name__, value)
        return value

def nop_write(prop):
    if False:
        i = 10
        return i + 15
    'Make this a property with a nop setter'

    def nop(self, value):
        if False:
            i = 10
            return i + 15
        pass
    return prop.setter(nop)