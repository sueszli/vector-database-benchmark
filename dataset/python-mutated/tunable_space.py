from .tunable_variable import Boolean, Choice, Fixed, FloatRange, IntRange

class TunableSpace:
    """
    A TunableSpace is constructed by the tunable variables.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._variables = {}
        self._values = {}

    @property
    def variables(self):
        if False:
            print('Hello World!')
        return self._variables

    @variables.setter
    def variables(self, variables):
        if False:
            i = 10
            return i + 15
        self._variables = variables

    @property
    def values(self):
        if False:
            for i in range(10):
                print('nop')
        return self._values

    @values.setter
    def values(self, values):
        if False:
            return 10
        self._values = values

    def get_value(self, name):
        if False:
            print('Hello World!')
        if name in self.values:
            return self.values[name]
        else:
            raise KeyError(f'{name} does not exist.')

    def set_value(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        if name in self.values:
            self.values[name] = value
        else:
            raise KeyError(f'{name} does not exist.')

    def _exists(self, name):
        if False:
            print('Hello World!')
        if name in self._variables:
            return True
        return False

    def _retrieve(self, tv):
        if False:
            i = 10
            return i + 15
        tv = tv.__class__.from_state(tv.get_state())
        if self._exists(tv.name):
            return self.get_value(tv.name)
        return self._register(tv)

    def _register(self, tv):
        if False:
            print('Hello World!')
        self._variables[tv.name] = tv
        if tv.name not in self.values:
            self.values[tv.name] = tv.default
        return self.values[tv.name]

    def __getitem__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.get_value(name)

    def __setitem__(self, name, value):
        if False:
            i = 10
            return i + 15
        self.set_value(name, value)

    def __contains__(self, name):
        if False:
            while True:
                i = 10
        try:
            self.get_value(name)
            return True
        except (KeyError, ValueError):
            return False

    def fixed(self, name, default):
        if False:
            for i in range(10):
                print('nop')
        tv = Fixed(name=name, default=default)
        return self._retrieve(tv)

    def boolean(self, name, default=False):
        if False:
            while True:
                i = 10
        tv = Boolean(name=name, default=default)
        return self._retrieve(tv)

    def choice(self, name, values, default=None):
        if False:
            for i in range(10):
                print('nop')
        tv = Choice(name=name, values=values, default=default)
        return self._retrieve(tv)

    def int_range(self, name, start, stop, step=1, default=None):
        if False:
            print('Hello World!')
        tv = IntRange(name=name, start=start, stop=stop, step=step, default=default)
        return self._retrieve(tv)

    def float_range(self, name, start, stop, step=None, default=None):
        if False:
            while True:
                i = 10
        tv = FloatRange(name=name, start=start, stop=stop, step=step, default=default)
        return self._retrieve(tv)

    def get_state(self):
        if False:
            print('Hello World!')
        return {'variables': [{'class_name': v.__class__.__name__, 'state': v.get_state()} for v in self._variables.values()], 'values': dict(self.values.items())}

    @classmethod
    def from_state(cls, state):
        if False:
            while True:
                i = 10
        ts = cls()
        for v in state['variables']:
            v = _deserialize_tunable_variable(v)
            ts._variables[v.name] = v
        ts._values = dict(state['values'].items())
        return ts

def _deserialize_tunable_variable(state):
    if False:
        return 10
    classes = (Boolean, Fixed, Choice, IntRange, FloatRange)
    cls_name_to_cls = {cls.__name__: cls for cls in classes}
    if isinstance(state, classes):
        return state
    if not isinstance(state, dict) or 'class_name' not in state or 'state' not in state:
        raise ValueError(f'Expect state to be a python dict containing class_name and state as keys, but found {state}')
    cls_name = state['class_name']
    cls = cls_name_to_cls[cls_name]
    if cls is None:
        raise ValueError(f'Unknown class name {cls_name}')
    cls_state = state['state']
    deserialized_object = cls.from_state(cls_state)
    return deserialized_object