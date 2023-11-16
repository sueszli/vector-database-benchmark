import random
import numpy as np
from nni.tuner import Tuner
from nni.utils import ClassArgsValidator

class DemoTuner(Tuner):

    def __init__(self, optimize_mode='maximize'):
        if False:
            while True:
                i = 10
        self.optimize_mode = optimize_mode

    def update_search_space(self, search_space):
        if False:
            i = 10
            return i + 15
        self._space = search_space

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            print('Hello World!')
        params = {}
        for k in self._space:
            (t, v) = (self._space[k]['_type'], self._space[k]['_value'])
            if t == 'choice':
                params[k] = random.choice(v)
            elif t == 'randint':
                params[k] = random.choice(range(v[0], v[1]))
            elif t == 'uniform':
                params[k] = np.random.uniform(v[0], v[1])
            else:
                raise RuntimeError('parameter type {} is supported by DemoTuner!'.format(t))
        return params

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            print('Hello World!')
        pass

class MyClassArgsValidator(ClassArgsValidator):

    def validate_class_args(self, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'optimize_mode' in kwargs:
            assert kwargs['optimize_mode'] in ['maximize', 'minimize'], 'optimize_mode {} is invalid!'.format(kwargs['optimize_mode'])