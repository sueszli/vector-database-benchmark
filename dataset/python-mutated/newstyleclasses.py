class NewStyleClassLibrary:

    def mirror(self, arg):
        if False:
            i = 10
            return i + 15
        arg = list(arg)
        arg.reverse()
        return ''.join(arg)

    @property
    def property_getter(self):
        if False:
            while True:
                i = 10
        raise SystemExit('This should not be called, ever!!!')

    @property
    def _property_getter(self):
        if False:
            while True:
                i = 10
        raise SystemExit('This should not be called, ever!!!')

class NewStyleClassArgsLibrary:

    def __init__(self, param):
        if False:
            while True:
                i = 10
        self.get_param = lambda self: param

class MyMetaClass(type):

    def __new__(cls, name, bases, ns):
        if False:
            print('Hello World!')
        ns['kw_created_by_metaclass'] = lambda self, arg: arg.upper()
        return type.__new__(cls, name, bases, ns)

    def method_in_metaclass(cls):
        if False:
            print('Hello World!')
        pass

class MetaClassLibrary(metaclass=MyMetaClass):

    def greet(self, name):
        if False:
            print('Hello World!')
        return 'Hello %s!' % name