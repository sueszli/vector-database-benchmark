import tensorflow as tf
import copy
from bigdl.nano.automl.utils import proxy_methods
from bigdl.nano.automl.tf.mixin import HPOMixin
from bigdl.nano.automl.hpo.space import AutoObject

@proxy_methods
class Sequential(HPOMixin, tf.keras.Sequential):
    """Tf.keras.Sequential with HPO capabilities."""

    def __init__(self, layers=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialzier.\n\n        :param layers: a list of layers (optional). Defults to None.\n        :param name: str(optional), name of the model. Defaults to None\n        '
        super().__init__(layers=None, name=name)
        self.model_class = tf.keras.Sequential
        self.name_ = name
        self.lazylayers_ = layers if layers is not None else []

    def add(self, layer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a layer.\n\n        :param layer: the layer to be added.\n        '
        self.lazylayers_.append(layer)

    def _model_init_args(self, trial):
        if False:
            print('Hello World!')
        instantiated_layers = []
        for layer in self.lazylayers_:
            if isinstance(layer, AutoObject):
                newl = self.backend.instantiate(trial, layer)
            else:
                newl = copy.deepcopy(layer)
            instantiated_layers.append(newl)
        return {'layers': instantiated_layers, 'name': self.name_}

    def _get_model_init_args_func_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the kwargs of _model_init_args_func except trial.'
        return {'lazylayers': self.lazylayers_, 'name': self.name_, 'backend': self.backend}

    @staticmethod
    def _model_init_args_func(trial, lazylayers, name, backend):
        if False:
            return 10
        instantiated_layers = []
        for layer in lazylayers:
            if isinstance(layer, AutoObject):
                newl = backend.instantiate(trial, layer)
            else:
                newl = copy.deepcopy(layer)
            instantiated_layers.append(newl)
        return {'layers': instantiated_layers, 'name': name}