class BiDi:
    """
    A wee utility class for keeping bi-directional mappings, like field
    constants in protocols. Names are attributes on the object, dict-like
    access maps values to names:

    CONST = BiDi(a=1, b=2)
    assert CONST.a == 1
    assert CONST.get_name(1) == "a"
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.names = kwargs
        self.values = {}
        for (k, v) in kwargs.items():
            self.values[v] = k
        if len(self.names) != len(self.values):
            raise ValueError('Duplicate values not allowed.')

    def __getattr__(self, k):
        if False:
            i = 10
            return i + 15
        if k in self.names:
            return self.names[k]
        raise AttributeError('No such attribute: %s', k)

    def get_name(self, n, default=None):
        if False:
            for i in range(10):
                print('nop')
        return self.values.get(n, default)