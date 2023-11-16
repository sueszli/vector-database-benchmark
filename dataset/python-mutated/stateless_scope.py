from keras.api_export import keras_export
from keras.backend.common import global_state

@keras_export('keras.StatelessScope')
class StatelessScope:

    def __init__(self, state_mapping=None, collect_losses=False, initialize_variables=True):
        if False:
            while True:
                i = 10
        from keras import backend
        from keras.backend.common.variables import KerasVariable
        self.collect_losses = collect_losses
        self.initialize_variables = initialize_variables
        self.losses = []
        self.state_mapping = {}
        state_mapping = state_mapping or {}
        for (k, v) in state_mapping:
            if not isinstance(k, KerasVariable):
                raise ValueError(f'Invalid reference variable in VariableSwapScope: all keys in argument `mapping` must be KerasVariable instances. Received instead: {k}')
            v = backend.convert_to_tensor(v, dtype=k.dtype)
            if k.shape != v.shape:
                raise ValueError(f'Invalid variable value in VariableSwapScope: all values in argument `mapping` must be tensors with a shape that matches the corresponding variable shape. For variable {k}, received invalid value {v} with shape {v.shape}.')
            self.state_mapping[id(k)] = v

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.original_scope = get_stateless_scope()
        global_state.set_global_attribute('stateless_scope', self)
        return self

    def add_loss(self, loss):
        if False:
            for i in range(10):
                print('nop')
        self.losses.append(loss)

    def add_update(self, update):
        if False:
            while True:
                i = 10
        (variable, value) = update
        self.state_mapping[id(variable)] = value

    def get_current_value(self, variable):
        if False:
            while True:
                i = 10
        return self.state_mapping.get(id(variable), None)

    def __exit__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        global_state.set_global_attribute('stateless_scope', self.original_scope)
        if self.original_scope is None and self.initialize_variables:
            from keras.backend.common.variables import initialize_all_variables
            initialize_all_variables()

def in_stateless_scope():
    if False:
        i = 10
        return i + 15
    return global_state.get_global_attribute('stateless_scope') is not None

def get_stateless_scope():
    if False:
        for i in range(10):
            print('nop')
    return global_state.get_global_attribute('stateless_scope')