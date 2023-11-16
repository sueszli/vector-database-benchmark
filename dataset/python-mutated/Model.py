import tensorflow as tf
from bigdl.nano.automl.utils import proxy_methods
from bigdl.nano.automl.tf.mixin import HPOMixin
from bigdl.nano.automl.hpo.callgraph import CallCache

@proxy_methods
class Model(HPOMixin, tf.keras.Model):
    """Tf.keras.Model with HPO capabilities."""

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        'Initializer.'
        super().__init__()
        self.model_class = tf.keras.Model
        self.kwargs = kwargs
        self.lazyinputs_ = kwargs.get('inputs', None)
        self.lazyoutputs_ = kwargs.get('outputs', None)

    def _model_init_args(self, trial):
        if False:
            print('Hello World!')
        (in_tensors, out_tensors) = CallCache.execute(self.lazyinputs_, self.lazyoutputs_, trial, self.backend)
        self.kwargs['inputs'] = in_tensors
        self.kwargs['outputs'] = out_tensors
        return self.kwargs

    def _get_model_init_args_func_kwargs(self):
        if False:
            print('Hello World!')
        'Return the kwargs of _model_init_args_func except trial.'
        return {'lazyinputs': self.lazyinputs_, 'lazyoutputs': self.lazyoutputs_, 'kwargs': self.kwargs, 'backend': self.backend}

    @staticmethod
    def _model_init_args_func(trial, lazyinputs, lazyoutputs, kwargs, backend):
        if False:
            for i in range(10):
                print('nop')
        (in_tensors, out_tensors) = CallCache.execute(lazyinputs, lazyoutputs, trial, backend)
        kwargs['inputs'] = in_tensors
        kwargs['outputs'] = out_tensors
        return kwargs