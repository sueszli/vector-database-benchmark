"""Autograph-specific overrides for ragged_tensor."""
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops.ragged import ragged_tensor

def _tf_ragged_for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    if False:
        i = 10
        return i + 15
    'Overload of for_stmt that iterates over TF ragged tensors.'
    init_vars = get_state()
    control_flow.verify_loop_init_vars(init_vars, symbol_names)
    if iter_.shape and iter_.shape[0] is not None:
        n = iter_.shape[0]
    else:
        n = iter_.row_lengths()[0]
    iterate_index = 0

    def aug_get_state():
        if False:
            for i in range(10):
                print('nop')
        return (iterate_index,) + get_state()

    def aug_set_state(aug_loop_vars):
        if False:
            i = 10
            return i + 15
        nonlocal iterate_index
        (iterate_index, *loop_vars) = aug_loop_vars
        set_state(loop_vars)

    def aug_body():
        if False:
            for i in range(10):
                print('nop')
        nonlocal iterate_index
        body(iter_[iterate_index])
        iterate_index += 1

    def aug_test():
        if False:
            while True:
                i = 10
        main_test = iterate_index < n
        if extra_test is not None:
            return tf_cond.cond(main_test, extra_test, lambda : False)
        return main_test
    control_flow._add_max_iterations_hint(opts, n)
    control_flow._tf_while_stmt(aug_test, aug_body, aug_get_state, aug_set_state, ('<internal iterate>',) + symbol_names, opts)
control_flow.for_loop_registry.register(ragged_tensor.RaggedTensor, _tf_ragged_for_stmt)