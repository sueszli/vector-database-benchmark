from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch

class AbstractLazyValue:

    def __init__(self, data, min=1, max=1):
        if False:
            print('Hello World!')
        self.data = data
        self.min = min
        self.max = max

    def __repr__(self):
        if False:
            return 10
        return '<%s: %s>' % (self.__class__.__name__, self.data)

    def infer(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

class LazyKnownValue(AbstractLazyValue):
    """data is a Value."""

    def infer(self):
        if False:
            print('Hello World!')
        return ValueSet([self.data])

class LazyKnownValues(AbstractLazyValue):
    """data is a ValueSet."""

    def infer(self):
        if False:
            print('Hello World!')
        return self.data

class LazyUnknownValue(AbstractLazyValue):

    def __init__(self, min=1, max=1):
        if False:
            i = 10
            return i + 15
        super().__init__(None, min, max)

    def infer(self):
        if False:
            return 10
        return NO_VALUES

class LazyTreeValue(AbstractLazyValue):

    def __init__(self, context, node, min=1, max=1):
        if False:
            print('Hello World!')
        super().__init__(node, min, max)
        self.context = context
        self._predefined_names = dict(context.predefined_names)

    def infer(self):
        if False:
            return 10
        with monkeypatch(self.context, 'predefined_names', self._predefined_names):
            return self.context.infer_node(self.data)

def get_merged_lazy_value(lazy_values):
    if False:
        return 10
    if len(lazy_values) > 1:
        return MergedLazyValues(lazy_values)
    else:
        return lazy_values[0]

class MergedLazyValues(AbstractLazyValue):
    """data is a list of lazy values."""

    def infer(self):
        if False:
            print('Hello World!')
        return ValueSet.from_sets((l.infer() for l in self.data))