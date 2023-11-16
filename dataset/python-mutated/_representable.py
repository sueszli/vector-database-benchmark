class _Representable:
    """This mix-in improves the default `__repr__` to be more readable."""

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = self.__dict__
        fields = [f'{key}={inputs[key]!r}' for key in inputs]
        fields = ', '.join(fields)
        return f'{self.__class__.__name__}({fields})'