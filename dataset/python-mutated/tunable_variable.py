import numpy as np

class TunableVariable:
    """
    Tunablevariable base class.
    """

    def __init__(self, name, default=None):
        if False:
            return 10
        self.name = name
        self._default = default

    @property
    def default(self):
        if False:
            print('Hello World!')
        return self._default

    def get_state(self):
        if False:
            i = 10
            return i + 15
        return {'name': self.name, 'default': self.default}

    @classmethod
    def from_state(cls, state):
        if False:
            while True:
                i = 10
        return cls(**state)

class Fixed(TunableVariable):
    """
    Fixed variable which cannot be changed.
    """

    def __init__(self, name, default):
        if False:
            return 10
        super().__init__(name=name, default=default)
        self.name = name
        if not isinstance(default, (str, int, float, bool)):
            raise ValueError(f'Fixed must be an str, int, float or bool, but found {default}')
        self._default = default

    def random(self, seed=None):
        if False:
            print('Hello World!')
        return self._default

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'Fixed(name: {self.name}, value: {self.default})'

class Boolean(TunableVariable):
    """
    Choice between True and False.
    """

    def __init__(self, name, default=False):
        if False:
            print('Hello World!')
        super().__init__(name=name, default=default)
        if default not in {True, False}:
            raise ValueError(f'default must be a Python boolean, but got {default}')

    def random(self, seed=None):
        if False:
            return 10
        rng = np.random.default_rng(seed)
        return rng.choice((True, False))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Boolean(name: "{self.name}", default: {self.default})'

class Choice(TunableVariable):

    def __init__(self, name, values, default=None):
        if False:
            return 10
        super().__init__(name=name, default=default)
        types = {type(v) for v in values}
        if len(types) > 1:
            raise TypeError('Choice can contain only one type of value, but found values: {} with types: {}.'.format(str(values), str(types)))
        self._is_unknown_type = False
        if isinstance(values[0], str):
            values = [str(v) for v in values]
            if default is not None:
                default = str(default)
        elif isinstance(values[0], int):
            values = [int(v) for v in values]
            if default is not None:
                default = int(default)
        elif isinstance(values[0], float):
            values = [float(v) for v in values]
            if default is not None:
                default = float(default)
        elif isinstance(values[0], bool):
            values = [bool(v) for v in values]
            if default is not None:
                default = bool(default)
        else:
            self._is_unknown_type = True
            self._indices = list(range(len(values)))
        self.values = values
        if default is not None and default not in values:
            raise ValueError('The default value should be one of the choices {}, but found {}'.format(values, default))
        self._default = default

    @property
    def default(self):
        if False:
            return 10
        if self._default is None:
            if None in self.values:
                return None
            return self.values[0]
        return self._default

    def random(self, seed=None):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(seed)
        if self._is_unknown_type:
            indice = rng.choice(self._indices)
            return self.values[indice]
        else:
            return rng.choice(self.values)

    def get_state(self):
        if False:
            i = 10
            return i + 15
        state = super().get_state()
        state['values'] = self.values
        return state

    def __repr__(self):
        if False:
            return 10
        return 'Choice(name: "{}", values: {}, default: {})'.format(self.name, self.values, self.default)

class IntRange(TunableVariable):
    """
    Integer range.
    """

    def __init__(self, name, start, stop, step=1, default=None, endpoint=False):
        if False:
            return 10
        super().__init__(name=name, default=default)
        self.start = self._check_int(start)
        self.stop = self._check_int(stop)
        self.step = self._check_int(step)
        self._default = default
        self.endpoint = endpoint

    @property
    def default(self):
        if False:
            for i in range(10):
                print('nop')
        if self._default is not None:
            return self._default
        return self.start

    def random(self, seed=None):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(seed)
        value = (self.stop - self.start) * rng.random() + self.start
        if self.step is not None:
            if self.endpoint:
                values = np.arange(self.start, self.stop + 1e-07, step=self.step)
            else:
                values = np.arange(self.start, self.stop, step=self.step)
            closest_index = np.abs(values - value).argmin()
            value = values[closest_index]
        return int(value)

    def get_state(self):
        if False:
            while True:
                i = 10
        state = super().get_state()
        state['start'] = self.start
        state['stop'] = self.stop
        state['step'] = self.step
        state['default'] = self._default
        return state

    def _check_int(self, val):
        if False:
            print('Hello World!')
        int_val = int(val)
        if int_val != val:
            raise ValueError(f'Expects val is an int, but found: {str(val)}.')
        return int_val

    def __repr__(self):
        if False:
            return 10
        return 'IntRange(name: {}, start: {}, stop: {}, step: {}, default: {})'.format(self.name, self.start, self.stop, self.step, self.default)

class FloatRange(TunableVariable):
    """
    Float range.
    """

    def __init__(self, name, start, stop, step=None, default=None, endpoint=False):
        if False:
            return 10
        super().__init__(name=name, default=default)
        self.stop = float(stop)
        self.start = float(start)
        if step is not None:
            self.step = float(step)
        else:
            self.step = None
        self._default = default
        self.endpoint = endpoint

    @property
    def default(self):
        if False:
            i = 10
            return i + 15
        if self._default is not None:
            return self._default
        return self.start

    def random(self, seed=None):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(seed)
        value = (self.stop - self.start) * rng.random() + self.start
        if self.step is not None:
            if self.endpoint:
                values = np.arange(self.start, self.stop + 1e-07, step=self.step)
            else:
                values = np.arange(self.start, self.stop, step=self.step)
            closest_index = np.abs(values - value).argmin()
            value = values[closest_index]
        return value

    def get_state(self):
        if False:
            print('Hello World!')
        state = super().get_state()
        state['start'] = self.start
        state['stop'] = self.stop
        state['step'] = self.step
        state['endpoint'] = self.endpoint
        return state

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'FloatRange(name: {}, start: {}, stop: {}, step: {}, default: {}, endpoint: {})'.format(self.name, self.start, self.stop, self.step, self.default, self.endpoint)