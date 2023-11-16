import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond

@export_case(example_inputs=(torch.ones(6),), tags={'torch.cond', 'torch.dynamic-shape'})
def cond_branch_nonlocal_variables(x):
    if False:
        print('Hello World!')
    '\n    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:\n      - both branches must take the same args, which must also match the branch args passed to cond.\n      - both branches must return a single tensor\n      - returned tensor must have the same tensor metadata, e.g. shape and dtype\n      - branch function can be free function, nested function, lambda, class methods\n      - branch function can not have closure variables\n      - no inplace mutations on inputs or global variables\n\n    This example demonstrates how to rewrite code to avoid capturing closure variables in branch functions.\n\n    The code below will not work because capturing closure variables is not supported.\n    ```\n    my_tensor_var = x + 100\n    my_primitive_var = 3.14\n\n    def true_fn(y):\n        nonlocal my_tensor_var, my_primitive_var\n        return y + my_tensor_var + my_primitive_var\n\n    def false_fn(y):\n        nonlocal my_tensor_var, my_primitive_var\n        return y - my_tensor_var - my_primitive_var\n\n    return cond(x.shape[0] > 5, true_fn, false_fn, [x])\n    ```\n\n    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.\n    '
    my_tensor_var = x + 100
    my_primitive_var = 3.14

    def true_fn(x, y, z):
        if False:
            print('Hello World!')
        return x + y + z

    def false_fn(x, y, z):
        if False:
            for i in range(10):
                print('nop')
        return x - y - z
    return cond(x.shape[0] > 5, true_fn, false_fn, [x, my_tensor_var, torch.tensor(my_primitive_var)])