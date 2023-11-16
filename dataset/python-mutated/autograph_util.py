"""Autograph utility functions for polymorphic_function."""
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.util import tf_decorator

def py_func_from_autograph(python_func, autograph_options=None):
    if False:
        while True:
            i = 10
    'Compile a python function using autograph, for use with FuncGraph.\n\n  Args:\n    python_func: the Python function to compile.\n    autograph_options: additional knobs to control when `autograph=True`.\n      See https://www.tensorflow.org/guide/autograph for more information.\n  Returns:\n    python_func, converted using autograph.\n  '
    (_, original_func) = tf_decorator.unwrap(python_func)

    def autograph_handler(*args, **kwargs):
        if False:
            print('Hello World!')
        'Calls a converted version of original_func.'
        try:
            return api.converted_call(original_func, args, kwargs, options=converter.ConversionOptions(recursive=True, optional_features=autograph_options, user_requested=True))
        except Exception as e:
            if hasattr(e, 'ag_error_metadata'):
                raise e.ag_error_metadata.to_exception(e)
            else:
                raise
    converted_func = tf_decorator.make_decorator(original_func, autograph_handler)
    return tf_decorator.rewrap(python_func, original_func, converted_func)