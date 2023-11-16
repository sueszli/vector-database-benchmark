class TemplateArg(str):
    """
    A cheetah template argument created from a param.
    The str of this class evaluates to the param's to code method.
    The use of this class as a dictionary (enum only) will reveal the enum opts.
    The __call__ or () method can return the param evaluated to a raw python data type.
    """

    def __new__(cls, param):
        if False:
            i = 10
            return i + 15
        value = param.to_code()
        instance = str.__new__(cls, value)
        setattr(instance, '_param', param)
        return instance

    def __getitem__(self, item):
        if False:
            return 10
        return str(self._param.get_opt(item)) if self._param.is_enum() else NotImplemented

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        if not self._param.is_enum():
            raise AttributeError()
        try:
            return str(self._param.get_opt(item))
        except KeyError:
            raise AttributeError()

    def __str__(self):
        if False:
            print('Hello World!')
        return str(self._param.to_code())

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._param.get_evaluated()