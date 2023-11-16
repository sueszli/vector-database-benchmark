class DynamicPositionalOnly:
    kws = {'one argument': ['one', '/'], 'three arguments': ['a', 'b', 'c', '/'], 'with normal': ['posonly', '/', 'normal'], 'default str': ['required', 'optional=default', '/'], 'default tuple': ['required', ('optional', 'default'), '/'], 'all args kw': [('one', 'value'), '/', ('named', 'other'), '*varargs', '**kwargs'], 'arg with separator': ['/one'], 'Too many markers': ['one', '/', 'two', '/'], 'After varargs': ['*varargs', '/', 'arg'], 'After named-only marker': ['*', '/', 'arg'], 'After kwargs': ['**kws', '/']}

    def get_keyword_names(self):
        if False:
            while True:
                i = 10
        return [key for key in self.kws]

    def run_keyword(self, name, args, kwargs=None):
        if False:
            print('Hello World!')
        if kwargs:
            return f'{name}-{args}-{kwargs}'
        return f'{name}-{args}'

    def get_keyword_arguments(self, name):
        if False:
            return 10
        return self.kws[name]