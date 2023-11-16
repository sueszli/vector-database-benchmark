class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        if False:
            i = 10
            return i + 15
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)

class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        if False:
            return 10
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if False:
            while True:
                i = 10
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if False:
            return 10
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if False:
            while True:
                i = 10
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if False:
        return 10
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)