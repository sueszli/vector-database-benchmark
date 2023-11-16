"""
Tool class to hold the param-space coordinates (X) and target values (Y).
"""
import numpy as np
import nni.parameter_expressions as parameter_expressions

def _hashable(params):
    if False:
        while True:
            i = 10
    '\n    Transform list params to tuple format. Ensure that an point is hashable by a python dict.\n\n    Parameters\n    ----------\n    params : numpy array\n        array format of parameters\n\n    Returns\n    -------\n    tuple\n        tuple format of parameters\n    '
    return tuple(map(float, params))

class TargetSpace:
    """
    Holds the param-space coordinates (X) and target values (Y)

    Parameters
    ----------
    pbounds : dict
        Dictionary with parameters names and legal values.

    random_state : int, RandomState, or None
        optionally specify a seed for a random number generator, by default None.
    """

    def __init__(self, pbounds, random_state=None):
        if False:
            while True:
                i = 10
        self._random_state = random_state
        self._keys = sorted(pbounds)
        self._bounds = np.array([item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])])
        for _bound in self._bounds:
            if _bound['_type'] == 'choice':
                try:
                    [float(val) for val in _bound['_value']]
                except ValueError:
                    raise ValueError('GP Tuner supports only numerical values')
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=0)
        self._cache = {}

    def __contains__(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        check if a parameter is already registered\n\n        Parameters\n        ----------\n        params : numpy array\n\n        Returns\n        -------\n        bool\n            True if the parameter is already registered, else false\n        '
        return _hashable(params) in self._cache

    def len(self):
        if False:
            print('Hello World!')
        '\n        length of registered params and targets\n\n        Returns\n        -------\n        int\n        '
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def params(self):
        if False:
            while True:
                i = 10
        '\n        registered parameters\n\n        Returns\n        -------\n        numpy array\n        '
        return self._params

    @property
    def target(self):
        if False:
            print('Hello World!')
        '\n        registered target values\n\n        Returns\n        -------\n        numpy array\n        '
        return self._target

    @property
    def dim(self):
        if False:
            while True:
                i = 10
        '\n        dimension of parameters\n\n        Returns\n        -------\n        int\n        '
        return len(self._keys)

    @property
    def keys(self):
        if False:
            while True:
                i = 10
        '\n        keys of parameters\n\n        Returns\n        -------\n        numpy array\n        '
        return self._keys

    @property
    def bounds(self):
        if False:
            return 10
        '\n        bounds of parameters\n\n        Returns\n        -------\n        numpy array\n        '
        return self._bounds

    def params_to_array(self, params):
        if False:
            i = 10
            return i + 15
        '\n        dict to array\n\n        Parameters\n        ----------\n        params : dict\n            dict format of parameters\n\n        Returns\n        -------\n        numpy array\n            array format of parameters\n        '
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError("Parameters' keys ({}) do ".format(sorted(params)) + 'not match the expected set of keys ({}).'.format(self.keys))
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        if False:
            i = 10
            return i + 15
        '\n        array to dict\n\n        maintain int type if the paramters is defined as int in search_space.json\n        Parameters\n        ----------\n        x : numpy array\n            array format of parameters\n\n        Returns\n        -------\n        dict\n            dict format of parameters\n        '
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError('Size of array ({}) is different than the '.format(len(x)) + 'expected number of parameters ({}).'.format(self.dim))
        params = {}
        for (i, _bound) in enumerate(self._bounds):
            if _bound['_type'] == 'choice' and all((isinstance(val, int) for val in _bound['_value'])):
                params.update({self.keys[i]: int(x[i])})
            elif _bound['_type'] in ['randint']:
                params.update({self.keys[i]: int(x[i])})
            else:
                params.update({self.keys[i]: x[i]})
        return params

    def register(self, params, target):
        if False:
            return 10
        '\n        Append a point and its target value to the known data.\n\n        Parameters\n        ----------\n        params : dict\n            parameters\n\n        target : float\n            target function value\n        '
        x = self.params_to_array(params)
        if x in self:
            print('Data point {} is not unique'.format(x))
        self._cache[_hashable(x.ravel())] = target
        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def random_sample(self):
        if False:
            return 10
        '\n        Creates a random point within the bounds of the space.\n\n        Returns\n        -------\n        numpy array\n            one groupe of parameter\n        '
        params = np.empty(self.dim)
        for (col, _bound) in enumerate(self._bounds):
            if _bound['_type'] == 'choice':
                params[col] = parameter_expressions.choice(_bound['_value'], self._random_state)
            elif _bound['_type'] == 'randint':
                params[col] = self._random_state.randint(_bound['_value'][0], _bound['_value'][1], size=1)
            elif _bound['_type'] == 'uniform':
                params[col] = parameter_expressions.uniform(_bound['_value'][0], _bound['_value'][1], self._random_state)
            elif _bound['_type'] == 'quniform':
                params[col] = parameter_expressions.quniform(_bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self._random_state)
            elif _bound['_type'] == 'loguniform':
                params[col] = parameter_expressions.loguniform(_bound['_value'][0], _bound['_value'][1], self._random_state)
            elif _bound['_type'] == 'qloguniform':
                params[col] = parameter_expressions.qloguniform(_bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self._random_state)
        return params

    def max(self):
        if False:
            print('Hello World!')
        '\n        Get maximum target value found and its corresponding parameters.\n\n        Returns\n        -------\n        dict\n            target value and parameters, empty dict if nothing registered\n        '
        try:
            res = {'target': self.target.max(), 'params': dict(zip(self.keys, self.params[self.target.argmax()]))}
        except ValueError:
            res = {}
        return res

    def res(self):
        if False:
            print('Hello World!')
        '\n        Get all target values found and corresponding parameters.\n\n        Returns\n        -------\n        list\n            a list of target values and their corresponding parameters\n        '
        params = [dict(zip(self.keys, p)) for p in self.params]
        return [{'target': target, 'params': param} for (target, param) in zip(self.target, params)]