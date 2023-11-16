"""Switch case for Control Flow Operations."""
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export

def _indexed_case_verify_and_canonicalize_args(branch_fns, default, branch_index):
    if False:
        print('Hello World!')
    'Verifies input arguments for the case function.\n\n  Args:\n    branch_fns: Dict or list of pairs of an `int` and a callable which returns a\n      list of tensors.\n    default: Optional callable that returns a list of tensors.\n    branch_index: Optional int `Tensor`, which selects for the corresponding\n      pred_fn_pair.\n\n  Raises:\n    TypeError: If `branch_fns` is not a list/dictionary.\n    TypeError: If `branch_fns` is a list but does not contain 2-tuples or\n               callables.\n    TypeError: If `fns[i]` is not callable for any i, or `default` is not\n               callable.\n\n  Returns:\n    branch_fns: validated list of callables for each branch (default last).\n  '
    if not isinstance(branch_index, tensor.Tensor):
        raise TypeError("'branch_index' must be a Tensor, got {}".format(type(branch_index)))
    if not branch_index.dtype.is_integer:
        raise TypeError("'branch_index' must be an integer Tensor, got {}".format(branch_index.dtype))
    if not branch_fns:
        raise ValueError("Must provide at least one item in 'branch_fns'")
    if not isinstance(branch_fns, (list, tuple, dict)):
        raise TypeError("'branch_fns' must be a list, tuple, or dict")
    if isinstance(branch_fns, dict):
        branch_fns = branch_fns.items()
    if all((callable(fn) for fn in branch_fns)):
        branch_fns = list(enumerate(branch_fns))
    for key_fn_pair in branch_fns:
        if not isinstance(key_fn_pair, tuple) or len(key_fn_pair) != 2:
            raise TypeError(f"Each entry in 'branch_fns' must be a 2-tuple. Received {key_fn_pair}.")
        (key, branch_fn) = key_fn_pair
        if not isinstance(key, int):
            raise TypeError('key must be a Python `int`, got {}'.format(type(key)))
        if not callable(branch_fn):
            raise TypeError('fn for key {} must be callable.'.format(key))
    keys = [p[0] for p in branch_fns]
    if min(keys) < 0 or max(keys) >= len(keys) or len(set(keys)) != len(keys):
        raise ValueError('branch indices (keys) must form contiguous range of [0 to {}) but found {{{}}}'.format(len(keys), ','.join(map(str, sorted(keys)))))
    actions = [p[1] for p in sorted(branch_fns)]
    if default is not None:
        actions.append(default)
    return actions

def _indexed_case_helper(branch_fns, default, branch_index, name, lower_using_switch_merge=None):
    if False:
        while True:
            i = 10
    'Implementation of case that emits the n-way indexed Case op.\n\n  Args:\n    branch_fns: Dict or list of pairs of a boolean scalar tensor, and a callable\n      which returns a list of tensors.\n    default: Optional callable that returns a list of tensors.\n    branch_index: Optional int `Tensor`, which selects for the corresponding\n      pred_fn_pair.\n    name: A name for this operation (optional).\n    lower_using_switch_merge: Lower this op using switch merge ops (optional).\n\n  Returns:\n    The tensors returned by the pair whose key matched branch_index, or\n    those returned by `default` if none does.\n\n  Raises:\n    TypeError: If `branch_fns` is not a list/dictionary.\n    TypeError: If `branch_fns` is a list but does not contain 2-tuples or\n               callables.\n    TypeError: If `fns[i]` is not callable for any i, or `default` is not\n               callable.\n  '
    branch_fns = _indexed_case_verify_and_canonicalize_args(branch_fns, default, branch_index)
    with ops.name_scope(name, 'case', [branch_index]):
        if context.executing_eagerly() and (not hasattr(branch_index, 'graph')):
            branch_index = array_ops.where(math_ops.less(branch_index, 0) | math_ops.greater_equal(branch_index, len(branch_fns)), len(branch_fns) - 1, branch_index)
            return branch_fns[int(branch_index)]()
        return cond_v2.indexed_case(branch_index, branch_fns, lower_using_switch_merge=lower_using_switch_merge)

@tf_export('__internal__.execute_fn_for_device', v1=[])
def execute_fn_for_device(device_branch_fns, default_fn, name='execute_fn'):
    if False:
        print('Hello World!')
    'Executes one of the provided callables based on the device placement.\n\n  This API is used when the implementations for high level function depend on\n  the underlying device placement. It takes a dictionary of device type to\n  callables. The device type includes "CPU", "GPU", "TPU", etc. When the type of\n  the device where to run this op matches the key in \'device_branch_fns\',\n  the corresponding callable is executed, falling back to \'default_fn\' if none\n  matches.\n\n  **Example:**\n  ```python\n  def f1(): return tf.constant(1)\n  def f2(): return tf.constant(2)\n  r = tf.execute_fn_for_device({"CPU": f1, "GPU": f2}, default_fn=f1)\n  ```\n  \'r\' is evaluated as 1 when it runs on CPU, 2 running on GPU, 1 running on\n  any other device types.\n\n\n  Args:\n    device_branch_fns: a dictionary of device types to the callables. Each\n      callable must return a matching structure of tensors.\n    default_fn: fallback callable when the underlying device does not match any\n      key in the \'device_branch_fns\'.\n    name: A name for this operation (optional).\n\n  Returns:\n    The tensors returned by the callable identified by device type during\n    execution, or those returned by \'default_fn\' if no key matches.\n  '
    is_in_xla = util.GraphOrParentsInXlaContext(ops.get_default_graph())
    if is_in_xla:
        return default_fn()
    device_branch_fns_upper = {k.upper(): v for (k, v) in device_branch_fns.items()}
    branch_fns = list(device_branch_fns_upper.values())
    devices = list(device_branch_fns_upper.keys())
    device_index = gen_functional_ops.device_index(device_names=devices)
    return _indexed_case_helper(branch_fns, default_fn, device_index, name, lower_using_switch_merge=False)

@tf_export('switch_case')
def switch_case(branch_index, branch_fns, default=None, name='switch_case'):
    if False:
        while True:
            i = 10
    'Create a switch/case operation, i.e.\n\n  an integer-indexed conditional.\n\n  See also `tf.case`.\n\n  This op can be substantially more efficient than `tf.case` when exactly one\n  branch will be selected. `tf.switch_case` is more like a C++ switch/case\n  statement than `tf.case`, which is more like an if/elif/elif/else chain.\n\n  The `branch_fns` parameter is either a dict from `int` to callables, or list\n  of (`int`, callable) pairs, or simply a list of callables (in which case the\n  index is implicitly the key). The `branch_index` `Tensor` is used to select an\n  element in `branch_fns` with matching `int` key, falling back to `default`\n  if none match, or `max(keys)` if no `default` is provided. The keys must form\n  a contiguous set from `0` to `len(branch_fns) - 1`.\n\n  `tf.switch_case` supports nested structures as implemented in `tf.nest`. All\n  callables must return the same (possibly nested) value structure of lists,\n  tuples, and/or named tuples.\n\n  **Example:**\n\n  Pseudocode:\n\n  ```c++\n  switch (branch_index) {  // c-style switch\n    case 0: return 17;\n    case 1: return 31;\n    default: return -1;\n  }\n  ```\n  or\n  ```python\n  branches = {0: lambda: 17, 1: lambda: 31}\n  branches.get(branch_index, lambda: -1)()\n  ```\n\n  Expressions:\n\n  ```python\n  def f1(): return tf.constant(17)\n  def f2(): return tf.constant(31)\n  def f3(): return tf.constant(-1)\n  r = tf.switch_case(branch_index, branch_fns={0: f1, 1: f2}, default=f3)\n  # Equivalent: tf.switch_case(branch_index, branch_fns={0: f1, 1: f2, 2: f3})\n  ```\n\n  Args:\n    branch_index: An int Tensor specifying which of `branch_fns` should be\n      executed.\n    branch_fns: A `dict` mapping `int`s to callables, or a `list` of (`int`,\n      callable) pairs, or simply a list of callables (in which case the index\n      serves as the key). Each callable must return a matching structure of\n      tensors.\n    default: Optional callable that returns a structure of tensors.\n    name: A name for this operation (optional).\n\n  Returns:\n    The tensors returned by the callable identified by `branch_index`, or those\n    returned by `default` if no key matches and `default` was provided, or those\n    returned by the max-keyed `branch_fn` if no `default` is provided.\n\n  Raises:\n    TypeError: If `branch_fns` is not a list/dictionary.\n    TypeError: If `branch_fns` is a list but does not contain 2-tuples or\n               callables.\n    TypeError: If `fns[i]` is not callable for any i, or `default` is not\n               callable.\n  '
    return _indexed_case_helper(branch_fns, default, branch_index, name)