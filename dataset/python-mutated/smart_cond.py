"""smart_cond and related utilities."""
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.smart_cond.smart_cond', v1=[])
def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    if False:
        print('Hello World!')
    'Return either `true_fn()` if predicate `pred` is true else `false_fn()`.\n\n  If `pred` is a bool or has a constant value, we return either `true_fn()`\n  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.\n\n  Args:\n    pred: A scalar determining whether to return the result of `true_fn` or\n      `false_fn`.\n    true_fn: The callable to be performed if pred is true.\n    false_fn: The callable to be performed if pred is false.\n    name: Optional name prefix when using `tf.cond`.\n\n  Returns:\n    Tensors returned by the call to either `true_fn` or `false_fn`.\n\n  Raises:\n    TypeError: If `true_fn` or `false_fn` is not callable.\n  '
    if not callable(true_fn):
        raise TypeError(f'Argument `true_fn` must be callable. Received {true_fn}')
    if not callable(false_fn):
        raise TypeError(f'Argument `false_fn` must be callable. Received {false_fn}')
    pred_value = smart_constant_value(pred)
    if pred_value is not None:
        if pred_value:
            return true_fn()
        else:
            return false_fn()
    else:
        return cond.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)

def smart_constant_value(pred):
    if False:
        print('Hello World!')
    'Return the bool value for `pred`, or None if `pred` had a dynamic value.\n\n  Args:\n    pred: A scalar, either a Python bool or tensor.\n\n  Returns:\n    True or False if `pred` has a constant boolean value, None otherwise.\n\n  Raises:\n    TypeError: If `pred` is not a Tensor or bool.\n  '
    if isinstance(pred, tensor.Tensor):
        pred_value = tensor_util.constant_value(pred)
        if pred_value is None:
            pred_value = tensor_util.try_evaluate_constant(pred)
    elif pred in {0, 1}:
        pred_value = bool(pred)
    elif isinstance(pred, bool):
        pred_value = pred
    else:
        raise TypeError(f'Argument `pred` must be a Tensor, or a Python bool, or 1 or 0. Received: pred={pred} of type {type(pred).__name__}')
    return pred_value

def smart_case(pred_fn_pairs, default=None, exclusive=False, name='smart_case'):
    if False:
        while True:
            i = 10
    'Like tf.case, except attempts to statically evaluate predicates.\n\n  If any predicate in `pred_fn_pairs` is a bool or has a constant value, the\n  associated callable will be called or omitted depending on its value.\n  Otherwise this functions like tf.case.\n\n  Args:\n    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a\n                   callable which returns a list of tensors.\n    default: Optional callable that returns a list of tensors.\n    exclusive: True iff at most one predicate is allowed to evaluate to `True`.\n    name: A name for this operation (optional).\n\n  Returns:\n    The tensors returned by the first pair whose predicate evaluated to True, or\n    those returned by `default` if none does.\n\n  Raises:\n    TypeError: If `pred_fn_pairs` is not a list/dictionary.\n    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.\n    TypeError: If `fns[i]` is not callable for any i, or `default` is not\n               callable.\n  '
    return control_flow_case._case_helper(smart_cond, pred_fn_pairs, default, exclusive, name, allow_python_preds=True)