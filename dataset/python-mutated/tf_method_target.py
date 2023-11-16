"""Module for the TFMethodTarget Class."""
import weakref
from tensorflow.python.util import tf_inspect

class TfMethodTarget:
    """Binding target for methods replaced by function and defun."""
    __slots__ = ('weakrefself_target__', 'weakrefself_func__')

    def __init__(self, target, original_python_function):
        if False:
            while True:
                i = 10
        self.weakrefself_target__ = target
        self.weakrefself_func__ = weakref.ref(original_python_function)

    @property
    def target(self):
        if False:
            print('Hello World!')
        return self.weakrefself_target__()

    @property
    def target_class(self):
        if False:
            while True:
                i = 10
        true_self = self.weakrefself_target__()
        if tf_inspect.isclass(true_self):
            return true_self
        else:
            return true_self.__class__

    def call(self, args, kwargs):
        if False:
            return 10
        wrapped_fn = self.weakrefself_func__()
        return wrapped_fn(self.weakrefself_target__(), *args, **kwargs)