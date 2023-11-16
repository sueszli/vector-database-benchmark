"""Autograph specifc overrides for dataset_ops."""
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest

def _general_purpose_scan(ds, init_state, body):
    if False:
        print('Hello World!')
    'Variant of Dataset.scan with semantics of general-purpose computation.'
    from tensorflow.python.data.ops import scan_op
    return scan_op._ScanDataset(ds, init_state, body, use_default_device=False)

def _tf_ag_dataset_for_stmt(ds, extra_test, body, get_state, set_state, symbol_names, opts):
    if False:
        print('Hello World!')
    'Overload of _dataset_for_stmt with early stopping. See for_stmt.'
    init_vars = get_state()
    control_flow.verify_loop_init_vars(init_vars, symbol_names)
    if not init_vars:
        init_vars = (constant_op.constant(0),)
        symbol_names = ('<internal dummy>',)

        def dummy_set_state(unused_dummy):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def dummy_get_state():
            if False:
                while True:
                    i = 10
            return (constant_op.constant(0),)
        (get_state, set_state) = (dummy_get_state, dummy_set_state)

    def scan_body(scan_state, scan_inputs):
        if False:
            for i in range(10):
                print('nop')
        'Main body of the Dataset.scan.'
        (loop_vars, iterate) = (scan_state, scan_inputs)
        set_state(loop_vars)

        def main_path():
            if False:
                print('Hello World!')
            body(iterate)
            new_loop_vars = get_state()
            control_flow.verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts, check_shapes=False)
            return new_loop_vars
        if extra_test is not None:
            extra_cond = extra_test()
            new_loop_vars = cond.cond(extra_cond, main_path, lambda : loop_vars)
        else:
            extra_cond = (constant_op.constant(True),)
            new_loop_vars = main_path()
        scan_outputs = (new_loop_vars, extra_cond)
        new_scan_state = new_loop_vars
        return (new_scan_state, scan_outputs)

    def take_while_predicate(unused_loop_vars, extra_cond):
        if False:
            print('Hello World!')
        return extra_cond

    def reduce_body(unused_reduce_state, scan_outputs):
        if False:
            print('Hello World!')
        (output_loop_vars, unused_extra_cond) = scan_outputs
        new_reduce_state = output_loop_vars
        return new_reduce_state
    ds = _general_purpose_scan(ds, init_vars, scan_body)
    if extra_test is not None:
        ds = ds.apply(take_while_ops.take_while(take_while_predicate))
    final_loop_vars = ds.reduce(init_vars, reduce_body)
    set_state(final_loop_vars)

def _tf_ag_dataset_abs(ds):
    if False:
        return 10
    specs = nest.flatten(ds.element_spec)
    if len(specs) == 1:
        return ds.map(math_ops.abs, num_parallel_calls=dataset_ops.AUTOTUNE)
    return ds.map(lambda *e: nest.map_structure(math_ops.abs, e), num_parallel_calls=dataset_ops.AUTOTUNE)

def _tf_ag_dataset_len(s):
    if False:
        for i in range(10):
            print('nop')
    'Autograph override of the builtin len for dataset_ops.DataSetV2.'
    l = s.cardinality()
    msg = gen_string_ops.string_join(['len requires dataset with definitive cardinality, got ', gen_string_ops.as_string(l)])
    with ops.control_dependencies([control_flow_assert.Assert(math_ops.logical_and(math_ops.not_equal(l, dataset_ops.INFINITE), math_ops.not_equal(l, dataset_ops.UNKNOWN)), [msg])]):
        l = array_ops.identity(l)
    return l

def _tf_ag_dataset_enumerate(ds, start=0):
    if False:
        print('Hello World!')
    return ds.enumerate(start)

def _tf_ag_dataset_zip(*iterables, strict=False):
    if False:
        for i in range(10):
            print('nop')
    if strict:
        raise ValueError('strict zip not supported by Dataset')
    return dataset_ops.DatasetV2.zip(iterables)

def _tf_ag_dataset_map(fn, *iterables):
    if False:
        return 10
    return dataset_ops.DatasetV2.zip(iterables).map(fn)

def _tf_ag_dataset_filter(fn, iterable):
    if False:
        for i in range(10):
            print('nop')
    return iterable.filter(fn)

def _tf_ag_dataset_any(iterable):
    if False:
        i = 10
        return i + 15
    specs = nest.flatten(iterable.element_spec)
    if len(specs) != 1 or specs[0].dtype != dtypes.bool:
        raise ValueError('in graph mode, the "any" builtin only supports datasets that return bool scalars; got: {}'.format(iterable.element_spec))
    ds = iterable.filter(lambda x: x)
    ds = ds.take(1)
    ds = ds.reduce(constant_op.constant(False, dtype=dtypes.bool), lambda _, y: y)
    return ds

def _tf_ag_dataset_all(iterable):
    if False:
        return 10
    specs = nest.flatten(iterable.element_spec)
    if len(specs) != 1 or specs[0].dtype != dtypes.bool:
        raise ValueError('in graph mode, the "all" builtin only supports datasets that return bool scalars; got: {}'.format(iterable.element_spec))
    ds = iterable.filter(math_ops.logical_not)
    ds = ds.take(1)
    ds = ds.reduce(constant_op.constant(True, dtype=dtypes.bool), lambda _, y: y)
    return ds

def register_overrides():
    if False:
        while True:
            i = 10
    'Registers the autograph specific overrides for dataset_ops.'
    control_flow.for_loop_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_for_stmt)
    py_builtins.abs_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_abs)
    py_builtins.len_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_len)
    py_builtins.enumerate_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_enumerate)
    py_builtins.zip_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_zip)
    py_builtins.map_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_map)
    py_builtins.filter_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_filter)
    py_builtins.any_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_any)
    py_builtins.all_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_all)