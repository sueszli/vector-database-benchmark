import functools

class ContextMethodDecorator:
    """A helper ContextManager decorating a method with a custom function."""

    def __init__(self, classx, method_name, decorator_func):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new context manager decorating a function within its scope.\n\n        This is a helper Context Manager that decorates a method of a class\n        with a custom function.\n        The decoration is only valid within the scope.\n        :param classx: A class (object)\n        :param method_name A string name of the method to be decorated\n        :param decorator_func: The decorator function is responsible\n         for calling the original method.\n         The signature should be: func(instance, original_method,\n         original_args, original_kwargs)\n         when called, instance refers to an instance of classx and the\n         original_method refers to the original method object which can be\n         called.\n         args and kwargs are arguments passed to the method\n\n        '
        self.method_name = method_name
        self.decorator_func = decorator_func
        self.classx = classx
        self.patched_by_me = False

    def __enter__(self):
        if False:
            print('Hello World!')
        self.original_method = getattr(self.classx, self.method_name)
        if not hasattr(self.original_method, 'sacred_patched%s' % self.__class__.__name__):

            @functools.wraps(self.original_method)
            def decorated(instance, *args, **kwargs):
                if False:
                    print('Hello World!')
                return self.decorator_func(instance, self.original_method, args, kwargs)
            setattr(self.classx, self.method_name, decorated)
            setattr(decorated, 'sacred_patched%s' % self.__class__.__name__, True)
            self.patched_by_me = True

    def __exit__(self, type, value, traceback):
        if False:
            i = 10
            return i + 15
        if self.patched_by_me:
            setattr(self.classx, self.method_name, self.original_method)