import inspect
inspect.formatargvalues(formatvalue=lambda value: __file__)
months = []
months.insert(0, lambda x: '')

class ExtendedInterpolation:

    def items(self, section, option, d):
        if False:
            print('Hello World!')
        value_getter = lambda option: self._interpolation.before_get(self, section, option, d[option], d)
        return value_getter

def test_Iterable(self):
    if False:
        return 10
    return (lambda : (yield))()