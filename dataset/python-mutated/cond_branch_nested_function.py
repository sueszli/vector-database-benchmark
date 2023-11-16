import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond

@export_case(example_inputs=(torch.ones(3),), tags={'torch.cond', 'torch.dynamic-shape'})
def cond_branch_nested_function(x):
    if False:
        return 10
    '\n    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:\n      - both branches must take the same args, which must also match the branch args passed to cond.\n      - both branches must return a single tensor\n      - returned tensor must have the same tensor metadata, e.g. shape and dtype\n      - branch function can be free function, nested function, lambda, class methods\n      - branch function can not have closure variables\n      - no inplace mutations on inputs or global variables\n\n    This example demonstrates using nested function in cond().\n\n    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.\n    '

    def true_fn(x):
        if False:
            while True:
                i = 10

        def inner_true_fn(y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y
        return inner_true_fn(x)

    def false_fn(x):
        if False:
            for i in range(10):
                print('nop')

        def inner_false_fn(y):
            if False:
                while True:
                    i = 10
            return x - y
        return inner_false_fn(x)
    return cond(x.shape[0] < 10, true_fn, false_fn, [x])