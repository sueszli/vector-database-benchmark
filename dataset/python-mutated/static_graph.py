import sys
import weakref
import numpy as np
import chainer
from chainer.backends import cuda
import chainer.function_node

def _is_xp(x):
    if False:
        print('Hello World!')
    return isinstance(x, np.ndarray) or isinstance(x, cuda.ndarray)

class ScheduleInfo(object):
    """A callable wrapper for a function in the static schedule.

    Args:
        func (FunctionNode): A function in the static schedule.
        args: Arguments to 'func'.
        kwargs: Keyword arguments to 'func'.
        inputs_hooks (list of tuples): A list of hooks that instruct how to
            update the ndarray references in 'args' so that they
            refer to the correct master array in 'unique_arrays'.
        return_hooks (list of tuples): A list of hooks that instruct how
            to update the ndarray references in 'unique_arrays' so that
            they refer to the correct arrays that were dynamically
            allocated and returned by 'func'. These run after
            'func' is called.
        unique_arrays (list of ndarray): The master list of all unique
            ndarrays that appear in the static schedule.
        func_name (str): An optional name of the static function. This is
            the name (if any) that was used as a decorater argument to
            `@static_code(func_name=name)`.
    """

    def __init__(self, func, args, kwargs, inputs_hooks, outputs_hooks, return_hooks, delete_hooks, unique_arrays, array_infos, func_name=None):
        if False:
            i = 10
            return i + 15
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.inputs_hooks = inputs_hooks
        self.outputs_hooks = outputs_hooks
        self.return_hooks = return_hooks
        self.unique_arrays = unique_arrays
        self.array_infos = array_infos
        assert len(self.array_infos) == len(self.unique_arrays)
        self.func_name = func_name
        self.in_list = None
        if self.inputs_hooks:
            self.in_list = self.kwargs['inputs']
        if self.outputs_hooks:
            self.out_list = self.kwargs['outputs']
        self.function_node = None
        if self.args:
            maybe_func = self.args[0]
            if isinstance(maybe_func, chainer.FunctionNode):
                self.function_node = maybe_func
        self.delete_hooks = delete_hooks

    def run_pre_hooks(self):
        if False:
            print('Hello World!')
        'Run hooks to set correct references.\n\n        This method is called from \'__call__()\'.\n        Process the list of hooks which will modify the array references in\n        the arguments list of the static function. This method must be\n        called before executing the static function.\n\n        The hooks specify that\n        each array argument points to a "master" array reference in the\n        unique_arrays list. If the reference in unique_arrays changes, then\n        we must also change the corresponding array reference in the arguments\n        list. The hooks specify the mapping and this method updates the\n        references in args to the corresponding values from unique_arrays.\n        '
        for hook in self.inputs_hooks:
            (ind, unique_ind) = hook
            self.in_list[ind] = self.unique_arrays[unique_ind]
        for hook in self.outputs_hooks:
            (ind, unique_ind) = hook
            self.out_list[ind] = self.unique_arrays[unique_ind]
        for ind in self.delete_hooks:
            self.unique_arrays[ind] = None

    def run_post_hooks(self, return_arrays):
        if False:
            while True:
                i = 10
        "Run post-hooks.\n\n        This method should be called after calling the static function\n        `self.func(*self.args)`. This method sets any array references that\n        appear in `self.args` to None. This is safe because the master\n        array reference is still kept in `self.unique_arrays`.\n\n        Also, process the list of post-hooks which will modify the array\n        references in\n        the unique_arrays list to refer to the new dynamically-allocated arrays\n        that were returned by 'func'.\n\n        Args:\n            return_arrays (list of ndarray or None): The list of arrays that\n                were returned by the schedule function, if not None.\n        "
        for hook in self.inputs_hooks:
            (ind, unique_ind) = hook
            self.in_list[ind] = None
        for hook in self.outputs_hooks:
            (ind, unique_ind) = hook
            self.out_list[ind] = None
        for hook in self.return_hooks:
            (ret_index, unique_list_index) = hook
            need_copy = self.array_infos[unique_list_index].retain
            if need_copy:
                self.unique_arrays[unique_list_index][...] = return_arrays[ret_index]
            else:
                self.unique_arrays[unique_list_index] = return_arrays[ret_index]

    def __call__(self):
        if False:
            print('Hello World!')
        self.run_pre_hooks()
        ret = self.func(*self.args, **self.kwargs)
        self.run_post_hooks(ret)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        out = 'function: ' + str(self.func) + '\n'
        out += 'name: ' + str(self.func_name) + '\n'
        out += 'args: ' + str(self.args) + '\n'
        out += 'kwargs: ' + str(self.args) + '\n'
        return out

class ArrayInfo(object):
    """Array information needed by the scheduler.

    This contains information about one array used in the naive static
    schedule corresponding to the define-by-run code.

    """

    def __init__(self, array):
        if False:
            i = 10
            return i + 15
        self.weak_ref = weakref.ref(array)
        self.id = id(array)
        self.array = None
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndarray_module = chainer.backend.get_array_module(array)
        if self.ndarray_module is cuda.cupy:
            self.device = cuda.get_device_from_array(array)
        else:
            self.device = -1
        self.in_var_index = None
        self.out_var_index = None
        self.dynamically_allocated = False
        self.dynamic_allocation_index = None
        self.dynamic_allocation_pass_depth = None
        self.dynamic_deletion_index = None
        self.dynamic_deletion_pass_depth = None
        self.static_allocation_index = None
        self.retain = False

    def was_deleted(self):
        if False:
            return 10
        return self.weak_ref() is None

    def get_new_empty_array(self):
        if False:
            return 10
        'Make and return a new empty ndarray.\n\n        Make and return a new empty ndarray that has the same shape,\n        dtype, and device as the array that was supplied to the\n        initializer.\n\n        '
        return self.ndarray_module.empty(self.shape, dtype=self.dtype)

    def __repr__(self):
        if False:
            while True:
                i = 10
        out = 'shape: {}\n'.format(self.shape)
        if self.was_deleted():
            out += 'Weak reference: dead\n'
        else:
            out += 'Weak reference: alive\n'
        if self.retain:
            out += 'Retained with retain_inputs()/retain_outputs().\n'
        if self.dynamically_allocated:
            out += 'Dynamically allocated at\n'
            out += '  pass_depth: {}\n'.format(self.dynamic_allocation_pass_depth)
            out += '  sched_index: {}\n'.format(self.dynamic_allocation_index)
        out += 'array id: {}'.format(self.id)
        return out

class StaticScheduleFunction(chainer.function_node.FunctionNode):
    """A function that executes the static schedule of a Chain.

    An instance of this class executes the static schedule of computations
    that are equivalent to executing the define-by-run code of a Chain.

    This class is used by the `static_graph` decorator to wrap the
    define-by-run
    computations of a chain into two static schedules:
    - The forward schedule corresponds to the computations that are executed by
    the define-by-run code of the `__call__()` method of a chain. The static
    schedule corresponding to these computations can be executed by calling the
    `forward()` method of this class.
    - The backward schedule corresponds to the computations that are executed
    by the sequence of calls to `Function.backward()` that occur during when
    backpropagating the gradients through the same chain. That is, for each
    `Function.forward()` that was called during the forward propagation,
    there will be a corresponding call to `Function.backward()` (of the
    same Function object) in the backward schedule. This backward schedule
    can be executed by calling the `backward()` method of this class.

    Note the intended usage of this class:

    Recall that a "static chain" referes to a chain that is decorated by the
    `static_graph` decorator.
    During the first forward pass of a static chain, the define-by-run code
    is executed. However,
    for subsequent iterations, that define-by-run code is replaced by an
    instance
    of this function and this function will be called instead. Since the
    static
    schedules contained by this function perform the same computations, it is
    safe (and potentially much more efficient) to simply execute the static
    schedule instead
    of the define-by-run code. See `static_graph` for details.

    Args:
        schedule_manager (ScheduleManager): The schedule manager of this
            schedule instance.
        in_vars (tuple of Variable): The flattened tuple of input variables
            that is supplied to
            `__call__()` method of the chain that this schedule corresponds to.
        unique_arrays (list of ndarray): A list of all unique array references
            deeply used in an StaticScheduleFunction instance. It is 'None'
            for the StaticScheduleFunction that corresponds to the "forward"
            schedule, but the contained StaticScheduleFunction for the
            "backward" schedule should take the unique_arrays of the
            "forward" schedule.

    """

    def __init__(self, schedule_manager, verbosity_level=0, enable_double_backprop=False):
        if False:
            while True:
                i = 10
        self.pass_depth = 0
        self.schedule_manager = schedule_manager
        self.schedule_info_list = []
        self.unique_arrays = []
        self.unique_array_infos = []
        self.array_id_to_unique_index = dict()
        self.backward_schedule_func = None
        self.verbosity_level = verbosity_level
        self.enable_double_backprop = enable_double_backprop
        self.in_vars = None
        self.chain = None
        self.schedule_built = False
        self.params_list = []
        self.grad_var_list = []
        self.array_id_to_param_map = dict()
        self.array_id_to_input_var_map = dict()
        self.param_id_to_index = dict()
        self.param_hooks = []
        self.param_post_hooks = []
        self.out_var_hooks = []
        self.in_var_hooks = []
        self.dynamically_allocated_unique_index = set()
        self.unique_ind_to_out_var_ind = dict()

    def get_unique_index_from_array(self, array):
        if False:
            print('Hello World!')
        'Return the array index if it exists.\n\n        Return the index of the array in self.unique_array_infos if the\n        array already exists in self.unique_array_info with a valid\n        reference. Otherwise, return None.\n        '
        ar_id = id(array)
        if ar_id in self.array_id_to_unique_index:
            unique_ind = self.array_id_to_unique_index[ar_id]
            info = self.unique_array_infos[unique_ind]
            assert ar_id == info.id
            if info.was_deleted():
                del self.array_id_to_unique_index[ar_id]
                return None
            else:
                return self.array_id_to_unique_index[ar_id]

    def get_contained_schedule(self):
        if False:
            print('Hello World!')
        sched = StaticScheduleFunction(self.schedule_manager, self.verbosity_level, self.enable_double_backprop)
        sched.pass_depth = self.pass_depth + 1
        sched.unique_arrays = self.unique_arrays
        sched.unique_array_infos = self.unique_array_infos
        sched.array_id_to_unique_index = self.array_id_to_unique_index
        sched.params_list = self.params_list
        sched.grad_var_list = self.grad_var_list
        sched.array_id_to_param_map = self.array_id_to_param_map
        sched.param_hooks = self.param_hooks
        sched.param_id_to_index = self.param_id_to_index
        return sched

    def is_empty(self):
        if False:
            i = 10
            return i + 15
        'Return True if this schedule is empty.\n\n        '
        return len(self.schedule_info_list) == 0

    def append_function(self, func, args, kwargs, func_name=None):
        if False:
            return 10
        "Append a function to the static schedule.\n\n        Append a function `func` to the static schedule. `func` can\n        be any function that is decorated with `@static_code` and that\n        was called while executing the static chain's `__call___()`\n        method, which contains the define-by-run code. The code\n        in the `@static_code` decorator will call this method to\n        add the function to the schedule just after it executes in\n        the define-by-run code as follows:\n\n        `return_arrays = func(*args, **kwargs)`\n\n        During the next iteration when the static chain switches from define-\n        by-run to the static schedule, a corresponding `ScheduleInfo`\n        object will call `func` as above, except that the scheduler might\n        make modifications\n        to some of the arrays in `kwargs` before and after the function is\n        called to implement various memory optimizations.\n\n        Args:\n            func (function or method): The function to append to the schedule.\n                This is a function that was decorated with `@static_code`.\n            args: The arguments that were originally supplied to `func` in\n                the define-by-run code of the static chain.\n            kwargs: The keyword arguments that were originally supplied to\n                `func` in the define-by-run code of the static chain.\n            func_name (str): Optional name for `func`, for debugging\n                purposes.\n            return_arrays (tuple of ndarray) or None: The value that is\n                returned by `func`, if any.\n\n        "
        retained_ids = set()
        last_sched_info_ind = len(self.schedule_info_list) - 1
        if last_sched_info_ind >= 0:
            prev_sched_info = self.schedule_info_list[last_sched_info_ind]
            if prev_sched_info.function_node is not None:
                retained_in_vars = prev_sched_info.function_node.get_retained_inputs()
                retained_out_vars = prev_sched_info.function_node.get_retained_outputs()
                if retained_in_vars is not None and retained_out_vars is not None:
                    retained_vars = retained_in_vars + retained_out_vars
                elif retained_in_vars is not None:
                    retained_vars = retained_in_vars
                elif retained_out_vars is not None:
                    retained_vars = retained_out_vars
                else:
                    retained_vars = None
                if retained_vars is not None:
                    for var in retained_vars:
                        retained_ids.add(id(var.data))
        for keep_id in retained_ids:
            unique_ind = self.array_id_to_unique_index[keep_id]
            array_info = self.unique_array_infos[unique_ind]
            array_info.retain = True
        delete_hooks = []
        for (unique_ind, ar_info) in enumerate(self.unique_array_infos):
            if ar_info.was_deleted():
                if ar_info.dynamic_deletion_index is None:
                    if self.verbosity_level >= 2:
                        print('Adding delete hook:')
                    delete_hooks.append(unique_ind)
                    ar_info.dynamic_deletion_index = last_sched_info_ind + 1
                    ar_info.dynamic_deletion_pass_depth = self.pass_depth
        ret = func(*args, **kwargs)
        inputs_hooks = []
        if 'inputs' in kwargs:
            in_list = kwargs['inputs']
            assert isinstance(in_list, list)
            for (ind, x) in enumerate(in_list):
                if _is_xp(x):
                    unique_ind = self.get_unique_index_from_array(x)
                    if unique_ind is None:
                        self.unique_arrays.append(None)
                        self.unique_array_infos.append(ArrayInfo(x))
                        unique_ind = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[id(x)] = unique_ind
                    inputs_hooks.append((ind, unique_ind))
                    in_list[ind] = None
        outputs_hooks = []
        if 'outputs' in kwargs:
            out_list = kwargs['outputs']
            assert isinstance(out_list, list)
            for (ind, x) in enumerate(out_list):
                if _is_xp(x):
                    unique_ind = self.get_unique_index_from_array(x)
                    if unique_ind is None:
                        self.unique_arrays.append(x)
                        self.unique_array_infos.append(ArrayInfo(x))
                        unique_ind = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[id(x)] = unique_ind
                    outputs_hooks.append((ind, unique_ind))
                    out_list[ind] = None
        return_hooks = []
        if ret is not None:
            assert isinstance(ret, list) or isinstance(ret, tuple)
            for (ret_index, item) in enumerate(ret):
                if _is_xp(item):
                    item_id = id(item)
                    unique_index = self.get_unique_index_from_array(item)
                    if unique_index is None:
                        self.unique_arrays.append(None)
                        ar_info = ArrayInfo(item)
                        ar_info.dynamically_allocated = True
                        sched_info_ind = len(self.schedule_info_list)
                        ar_info.dynamic_allocation_index = sched_info_ind
                        ar_info.dynamic_allocation_pass_depth = self.pass_depth
                        self.unique_array_infos.append(ar_info)
                        unique_index = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[item_id] = unique_index
                    else:
                        unique_index = self.array_id_to_unique_index[item_id]
                        print('the current id: ', item_id)
                        print('the unique_index: ', unique_index)
                        print('array info: ', self.unique_array_infos[unique_ind])
                        raise RuntimeError('Found result array from schedule function already in unique_arrays!')
                    return_hooks.append((ret_index, unique_index))
                    self.dynamically_allocated_unique_index.add(unique_index)
        if self.verbosity_level >= 2:
            print('Adding function to static schedule: ', func)
        self.schedule_info_list.append(ScheduleInfo(func, args, kwargs, inputs_hooks, outputs_hooks, return_hooks, delete_hooks, self.unique_arrays, self.unique_array_infos, func_name=func_name))
        return ret

    def __repr__(self):
        if False:
            return 10
        out = 'StaticSchedule:\n'
        if self.pass_depth == 0:
            depth = 'forward pass'
        elif self.pass_depth == 1:
            depth = 'backward pass'
        elif self.pass_depth == 2:
            depth = 'double backward pass'
        else:
            depth = str(self.pass_depth)
        out += 'Pass depth: ' + depth + '\n'
        out += 'Length of unique_arrays: ' + str(len(self.unique_arrays)) + '\n'
        for x in self.schedule_info_list:
            out += str(x)
        return out

    def debug_print_ref_counts(self):
        if False:
            return 10
        print('reference counts in unique_arrays:')
        for ind in range(len(self.unique_arrays)):
            print('index: ', ind)
            print('reference count: ', sys.getrefcount(self.unique_arrays[ind]))

    def run_param_pre_hooks(self):
        if False:
            i = 10
            return i + 15
        "Run parameter reference updater hooks.\n\n        It also handles the case where the 'grad' attribute\n        was set to 'None' by outside Chainer code.\n\n        "
        for hook in self.param_hooks:
            (unique_array_index, param_attribute_location) = hook
            (params_list_index, attribute_location) = param_attribute_location
            if attribute_location == 'data':
                self.unique_arrays[unique_array_index] = self.params_list[params_list_index].data
            elif attribute_location == 'grad':
                self.params_list[params_list_index].grad = self.unique_arrays[unique_array_index]

    def run_param_post_hooks(self):
        if False:
            while True:
                i = 10
        'Update parameter attributes after schedule is executed.\n\n        If any dynamically-allocated arrays in the schedule correspond to\n        a parameter attribute, it must be updated after the schedule is\n        run.\n        '
        if self.verbosity_level >= 2:
            print('run_param_post_hooks()...')
        for hook in self.param_post_hooks:
            (unique_array_index, param_attribute_location) = hook
            (params_list_index, attribute_location) = param_attribute_location
            if attribute_location == 'data':
                self.params_list[params_list_index].data = self.unique_arrays[unique_array_index]
            elif attribute_location == 'grad':
                self.params_list[params_list_index].grad = self.unique_arrays[unique_array_index]

    def run_in_var_hooks(self, input_var_arrays):
        if False:
            for i in range(10):
                print('nop')
        "Run hooks to update variable array references.\n\n        Args:\n            input_var_arrays (tuple of ndarray): The 'data' array attributes\n                of the input variables to this function.\n        "
        for hook in self.in_var_hooks:
            (unique_array_index, in_var_ind) = hook
            if self.verbosity_level >= 2:
                print('input var hook:')
                print('unique_array_index: ', unique_array_index)
                print('in_var_ind: ', in_var_ind)
                print('_run_in_var_hooks(): Using this input variable array for forward pass: ', input_var_arrays[in_var_ind])
            self.unique_arrays[unique_array_index] = input_var_arrays[in_var_ind]

    def debug_print_unique_arrays_info(self):
        if False:
            i = 10
            return i + 15
        for (ind, item) in enumerate(self.unique_arrays):
            print('--- unique_arrays ---')
            print('index: {0}; id: {1}'.format(ind, id(item)))
            if item is not None:
                print('shape: ', item.shape)
            if ind in self.unique_ind_to_out_var_ind:
                out_var_ind = self.unique_ind_to_out_var_ind[ind]
                print('output variable at return index: ', out_var_ind)
            if ind in self.dynamically_allocated_unique_index:
                print('Dynamically allocated inside schedule.')

    def run_out_var_hooks(self):
        if False:
            print('Hello World!')
        'Run hooks to update output variable array references.\n\n\n        '
        for hook in self.out_var_hooks:
            (out_var_ind, unique_list_index) = hook
            out_var = self.out_vars[out_var_ind]
            out_var.data = self.unique_arrays[unique_list_index]
            if self.verbosity_level >= 2:
                print('StaticScheduleFunction: running output variable hook: out_var_ind, unique_list_index): ', hook)

    def set_out_variables(self, out_vars):
        if False:
            for i in range(10):
                print('nop')
        "Set output variables.\n\n        This should be called after the define-by-run code in the\n        chain's `__call__()` has already run but before running the\n        static schedule.\n\n        Args:\n            out_vars (list of Variable): The (flattened) list of output\n                variables obtained by performing a define-by-run\n                forward pass (or corresponding backward pass) on the\n                local sub-graph corresponding to the static chain.\n        "
        self.out_vars = out_vars
        for (var_ind, var) in enumerate(out_vars):
            if var is not None:
                key = id(var.data)
                if key in self.array_id_to_unique_index:
                    unique_list_index = self.array_id_to_unique_index[key]
                    self.out_var_hooks.append((var_ind, unique_list_index))
                    self.unique_ind_to_out_var_ind[unique_list_index] = var_ind
                else:
                    raise RuntimeError('Could not find output variable in unique_arrays.')

    def build_schedule(self, chain, in_vars):
        if False:
            return 10
        "Build the static schedule.\n\n        Perform one-time post-processing on the functions and arguments\n        that were\n        previously supplied in 'append_function()' to create the static\n        schedule.\n\n        This method must be called after the final call of 'append_function()'\n        and before calling 'forward()' for the first time.\n\n        Args:\n            chain: The static chain that uses this scheudle.\n            in_vars (list of Variable): The input variables to this static\n                schedule. This are the input variables (each having no\n                creator) of the local sub-graph corresponding to the\n                static chain.\n        "
        self.chain = chain
        self.in_vars = in_vars
        if self.verbosity_level >= 2:
            print('Building schedule for pass depth: ', self.pass_depth)
        for (ind, info) in enumerate(self.unique_array_infos):
            if self.verbosity_level >= 2:
                print('unique array index: ', ind)
                print('array info: ', info)
            if not info.was_deleted():
                self.unique_arrays[ind] = info.weak_ref()
        unique_ids = set()
        for ar in self.unique_arrays:
            if ar is not None:
                assert id(ar) not in unique_ids
                unique_ids.add(id(ar))
        for param in chain.params():
            param_key = id(param)
            if param_key not in self.param_id_to_index:
                self.params_list.append(param)
                grad_var = param.grad_var
                self.grad_var_list.append(grad_var)
                param_index = len(self.params_list) - 1
                self.param_id_to_index[param_key] = param_index
            else:
                param_index = self.param_id_to_index[param_key]
            grad_var = param.grad_var
            self.grad_var_list[param_index] = grad_var
            if param.data is not None:
                key = id(param.data)
                if key not in self.array_id_to_param_map:
                    self.array_id_to_param_map[key] = (param_index, 'data')
            if param.grad is not None:
                key = id(param.grad)
                if key not in self.array_id_to_param_map:
                    self.array_id_to_param_map[key] = (param_index, 'grad')
        for (var_ind, in_var) in enumerate(self.in_vars):
            assert in_var.data is not None
            key = id(in_var.data)
            self.array_id_to_input_var_map[key] = var_ind
        assert len(self.unique_arrays) > 0
        for (unique_array_index, ar) in enumerate(self.unique_arrays):
            key = id(ar)
            if key in self.array_id_to_param_map:
                param_attribute_location = self.array_id_to_param_map[key]
                param_hook = (unique_array_index, param_attribute_location)
                self.param_hooks.append(param_hook)
            if key in self.array_id_to_input_var_map:
                in_var_ind = self.array_id_to_input_var_map[key]
                in_var_hook = (unique_array_index, in_var_ind)
                self.in_var_hooks.append(in_var_hook)
                if self.verbosity_level >= 2:
                    print('build_schedule(): Adding input variable hook: ', in_var_hook)
                    print('For input variable: ', ar)
            if unique_array_index in self.dynamically_allocated_unique_index:
                if key in self.array_id_to_param_map:
                    param_attribute_location = self.array_id_to_param_map[key]
                    param_hook = (unique_array_index, param_attribute_location)
                    self.param_post_hooks.append(param_hook)
        if self.verbosity_level >= 2:
            print('self.param_hooks: ', self.param_hooks)
            self.debug_print_unique_arrays_info()
        print('end of build_schedule()')
        self.schedule_built = True

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        if self.verbosity_level >= 2:
            print('Calling StaticScheduleFunction.forward()...')
        if not self.schedule_built:
            raise RuntimeError('forward() was called before build_schedule()!')
        self.run_param_pre_hooks()
        self.run_in_var_hooks(inputs)
        if self.verbosity_level >= 2:
            print('Running static schedule...')
        for x in self.schedule_info_list:
            x()
        if self.verbosity_level >= 2:
            self.debug_print_unique_arrays_info()
        self.run_out_var_hooks()
        self.run_param_post_hooks()
        ret = []
        for y in self.out_vars:
            if y is None or y.data is None:
                ret.append(None)
            else:
                ret.append(y.data.copy())
        return tuple(ret)

    def backward(self, target_input_indexes, grad_outputs):
        if False:
            return 10
        if self.verbosity_level >= 2:
            print('Calling StaticScheduleFunction.backward()...')
        self.schedule_manager.end_forward()
        if self.backward_schedule_func is None:
            print('Creating new backward schedule...')
            self.backward_schedule_func = self.get_contained_schedule()
            new_grad_outputs = []
            for var in grad_outputs:
                new_grad_outputs.append(chainer.Variable(var.data))
            with chainer.using_config('schedule_func', self.backward_schedule_func):
                with chainer.using_config('enable_backprop', True):
                    for (ind, var) in enumerate(new_grad_outputs):
                        self.out_vars[ind].grad = new_grad_outputs[ind].data
                    inputs = [param for param in self.chain.params()]
                    for var in self.in_vars:
                        inputs.append(var)
                    ugh = self.enable_double_backprop
                    chainer.grad(self.out_vars, inputs, grad_outputs=new_grad_outputs, set_grad=True, enable_double_backprop=ugh)
            backward_out_vars = [var.grad_var for var in self.in_vars]
            self.backward_schedule_func.set_out_variables(backward_out_vars)
            for n in range(len(self.in_vars)):
                self.in_vars[n] = None
            if self.verbosity_level >= 2:
                print('building backward schedule.')
            self.backward_schedule_func.build_schedule(self.chain, new_grad_outputs)
        return self.backward_schedule_func.apply(grad_outputs)

class ScheduleManager(object):
    """A manager of static schedules for a static chain.

    This is a container of the static schedules that are used by a static
    chain.

    Args:
        minimize_cache_size (bool): If `True`, attempt to reduce memory
        usage by clearing the cached schedules whenever the training
        mode changes (that is, whenever `chainer.config.train` changes
        value) or whenever the mini-batch size changes.


    """

    def __init__(self, minimize_cache_size=True, verbosity_level=0):
        if False:
            while True:
                i = 10
        self.schedules = dict()
        self.minimize_cache_size = minimize_cache_size
        self.in_use_count = dict()
        self.forward_over = False
        self.prev_train_config = None
        self.max_in_use_train = 0
        self.train_count = 0
        self.verbosity_level = verbosity_level

    def get_schedule(self, in_vars, enable_double_backprop=False):
        if False:
            i = 10
            return i + 15
        'Get a static schedule.\n\n        Return a static schedule object (that is, an instance of\n        ``StaticScheduleFunction``) that is compatible with\n        the current configuration and input variables to the supplied chain.\n        If there is no existing schedule available, return an empty schedule\n        object.\n\n        During the usual "training mode" (that is, when both\n        `chainer.config.enable_backprop` and `chainer.config.train`\n        are `True`), this method will always return a distince static\n        schedule each time it is called within the same iteration.\n        It will also try to reuse\n        existing schedules across iterations. Therefore, any schedule that\n        is returned in a given iteration cannot be returned again until\n        the following iteration. However, if either of these flags is\n        \'False\', then this method may return the same schedule instance\n        multiple times within the same iteration, as long as it is\n        compatible with `in_vars`.\n\n        Note that in order to implement the above behavior, the schedule\n        manager must be informed when the current iteration has finished.\n        This is accomplished by calling `end_forward()` after the\n        iteration has finished. If a backward pass is performed, then\n        `end_forward()` will be automatically called. Otherwise, it\n        will not be called and the user will be responsible for calling\n        it.\n\n        Args:\n            in_vars (tuple of :class:`~chainer.Variable`): The input\n                variables to the chain.\n\n        Returns:\n            An instance of ``StaticScheduleFunction``.\n        '
        if self.forward_over:
            self.forward_over = False
        if self.minimize_cache_size:
            if chainer.config.train != self.prev_train_config:
                self.prev_train_config = chainer.config.train
                if self.verbosity_level >= 2:
                    print('Clearing schedule cache...')
                self.schedules.clear()
                self.in_use_count.clear()
        if chainer.config.train is False or chainer.config.enable_backprop is False:
            key_str = 'test:' + ''.join((str(x.shape) + str(x.dtype) for x in in_vars))
            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                sched = sched_list[0]
            else:
                vb = self.verbosity_level
                edb = enable_double_backprop
                sched = StaticScheduleFunction(self, verbosity_level=vb, enable_double_backprop=edb)
                self.schedules[key_str] = [sched]
            return sched
        else:
            key_str = 'train:' + ''.join((str(x.shape) + str(x.dtype) for x in in_vars))
            self.train_count += 1
            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                available_index = self.in_use_count[key_str]
                if available_index >= len(sched_list):
                    vb = self.verbosity_level
                    edb = enable_double_backprop
                    sched = StaticScheduleFunction(self, verbosity_level=vb, enable_double_backprop=edb)
                    sched_list.append(sched)
                sched = sched_list[available_index]
                self.in_use_count[key_str] = available_index + 1
            else:
                vb = self.verbosity_level
                edb = enable_double_backprop
                sched = StaticScheduleFunction(self, verbosity_level=vb, enable_double_backprop=edb)
                self.schedules[key_str] = [sched]
                self.in_use_count[key_str] = 1
        return sched

    def end_forward(self):
        if False:
            while True:
                i = 10
        'Make in-use schedules available for use in next iteration.\n\n        Set the in-use status of all schedules to "not in use" so that\n        they can be reused in the next iteration.\n\n        In the case that test mode is active\n        (`chainer.config.train` is `False`) and the static chain corresponding\n        to this manager was not called more than once in any iteration during\n        training mode, then this method will be called automatically.\n\n        '
        if not self.forward_over:
            for key in self.in_use_count:
                self.in_use_count[key] = 0
            self.forward_over = True
            if self.train_count > self.max_in_use_train:
                self.max_in_use_train = self.train_count
                if self.verbosity_level >= 2:
                    print('Maximum in-use schedules per training iteration: ', self.max_in_use_train)
            self.train_count = 0

    def __repr__(self):
        if False:
            while True:
                i = 10
        out = 'ScheduleManager:\n'
        for key_str in self.schedules:
            out += 'key string: ' + key_str
            sched_list = self.schedules[key_str]
            out += ' -> schedule list of length: ' + str(len(sched_list)) + '\n'
            for sched in sched_list:
                out += str(sched)
        return out

def static_graph(*args, **kwargs):
    if False:
        print('Hello World!')
    'Decorator to mark a Chain\'s ``__call__()`` as a static sub-graph.\n\n    This decorator marks the define-by-run code inside the `__call__()`\n    method of a Chain instance as corresponding to a static computation\n    graph or sub-graph. Such a chain will be referred to as a \'static chain\'.\n    This allows various "static graph" optimizations to be performed, which\n    can result in significant speedups for some models.\n\n    When this decorator is used, the chain\'s define-by-run code executes\n    during the first iteration as usual. However, while the define-by-run\n    code is executing, a trace is also performed to incrementally create a\n    corresponding static schedule. This static schedule will only contain\n    the subset of the computations inside the define-by-run code that actually\n    needs to run every iteration. Specifically, this will contain the code\n    inside any functions called that were annotated with the `@static_code`\n    decorator, which will include all Chainer built-in functions, as well as\n    any user-defined functions that use `@static_code`. Then, starting\n    from the second iteration, when the static chain is called, its\n    static schedule code will be executed instead of its define-by-run code.\n\n    However, the user must also be careful of the following:\n    - The user is responsible for applying this decorator correctly. The\n    framework\n    does not check that the define-by-run code corresponds to a static\n    graph. The graph can be different between training and\n    evaluation mode (such as when dropout and/or batch normalization are\n    used), but should otherwise be static.\n    - When `chainer.config.enable_backprop` is enabled, if a backward pass\n    is not performed each iteration, then the user code must call a method\n    `chain.schedule_manager.end_forward()`on the static chain each iteration.\n    - Static graphs allow tradeoffs between computation and memory usage.\n    For example, the `minimize_cache_size` argument will typically result in\n    higher memory usage when set to `False` because all cached schedules\n    are retained.\n    - When this feature is enabled, only the Chainer function and/or link\n    calls inside the chain\'s `__call__()` method will be included in the\n    static schedule by default. An other code that the user puts in\n    `__call__()`, such as a print statement or code to increment a counter\n    for example, will not automatically get added. We will refer to such\n    code other than Chainer function/link calls as "side-effect" code.\n    Since side-effect code does not get included in the static schedule\n    by default, this means that it will only every execute once, during\n    the first iteration. There is a way to force side-effect code to be\n    included in the static schedule, however: the user can wrapp such\n    code inside a function that is decorated with\n    `@static_code` to ensure that it gets added to the static schedule.\n    For an example of this, refer to the documentation.\n    - This feature is experimental and advanced optimizations such\n    as kernel fusion and various memory optimizations are not implemented\n    yet.\n\n    Usage:\n\n    This decorator should only be applied\n    to define-by-run code that actually corresponds to a static subgraph.\n    Refer to the documenation for additional details and examples of\n    correct usage.\n    This decorator should be applied to each of the largest static\n    subgraphs in the model; it can also be applied to a static subgraph\n    that is not the largest subgraph, but that could result in reduced\n    performance.\n    It is not currently allowed to\n    mark a chain as static if it is contained within\n    another chain that is also marked as being static.\n    For example, suppose a\n    static graph `A` contains a static sub-graph `B`. Then, only the chain\n    corresponding to `A` should be marked as static and the chain\n    corresponding\n    to `B` should not be marked as static.\n\n    The behavior of a static chain depends on the training mode flag,\n    `chainer.config.train`. If it is `True`, then a static chain that is\n    called multiple times will try to use a distinct static schedule object\n    (that is, call a distinct instance of a FunctionNode that implements\n    that static schedule) on each call. The same schedule instance cannot\n    be reused until the forward pass has completed, which is signaled by\n    performing a backward pass through the model. It is therefore important\n    that the backward pass be performed after each forward pass during\n    training. Since this is usually the case, most usages of static chain\n    will not required any modifications to existing code other than applying\n    this decorator. However, if you would like to perform multiple forward\n    passes during training before performing a backward pass, then you\n    must call `chain.schedule_manager.end_forward()` after the end\n    of each forward pass.\n\n    If test mode is active (`chainer.config.train` is `False`) then it\n    is not necessary to inform the chain at the end of each forward pass\n    because in test mode, a static chain always attempts to reuse\n    existing static schedule objects. The same static schedule can be reused\n    during a single forward pass, because it is not necessary to compute\n    gradients.\n    It is also possible to disable static optimzations while in test mode by\n    setting the decorator argument `force_test_define_by_run=True`.\n\n    Note: If either \'chainer.config.enable_backprop\' or \'chainer.config.train\'\n    is set to \'False\', then cached static schedules will be reused when\n    possible to reduce memory usage.\n\n    Double-backprop:\n        Double-backpropagation is not enabled by default. It can be enabled by\n        supplying the keyword argument ``enable_double_backprop=True``\n        to this decorator. Note: this feature has not been tested yet.\n\n    Restrictions on input arguments and return values of a static chain:\n        Recall that unlike a function, there is no restrictions on the\n        arguments to a chain. However, there currently are some restrictions\n        when a static chain is used. Specifically, the arguments to a static\n        chain must consist of a variable, list or tuple. In the case of a list\n        or tuple, the elements are required to be an instance of variable,\n        list, or tuple. There can be an arbitrary number of nested lists/\n        tuples. No other object types are allowed.\n        In addition, keyword arguments are not allowed.\n        The return value of a static chain must be a\n        variable, list, or tuple in which each element of the list or\n        tuple is also a variable, list, or tuple.\n\n    This decorator can be supplied with the following optional keyword\n    arguments. This is an experimental feature, and the API and arguments\n    might change\n\n    Args:\n        force_test_define_by_run (bool): If `True`, disable static graph\n            optimizations during test mode (that is, when\n            `chainer.config.train` is False). This may be needed in order\n            for some existing RNN links such as LSTM to work correctly,\n            since some existing links do not correspond to a static graph\n            in some cases.\n            The default is `False`.\n\n        minimize_cache_size (bool): If `True`, minimize the number of cached\n            static schedules in order to reduce memory usage. For example,\n            if the mini-batch size changes or the training mode changes,\n            the schedules will need to be recomputed, but memory is also\n            saved by not retaining all cached schedules.\n            The default value is `True`.\n\n        verbosity_level (int): Depending on the value, print additional\n            information:\n            0: Warnings only. (the default value)\n            1: Show only information that is collected during the first\n            iteration and when a new static schedule is created.\n            2: Detailed debugging information, possibly showing new\n            information every iteration.\n\n        enable_double_backprop (bool): If `True`, enable double-backprop.\n            The default value is `False` (not enabled).\n\n    Returns:\n        Wrapped ``__call__()`` method with static chain support.\n\n    '
    force_test_define_by_run = False
    minimize_cache_size = False
    verbosity_level = 0
    enable_double_backprop = False
    zero_args = False
    if len(args) == 1 and (not kwargs) and callable(args[0]):
        callable_arg = args[0]
        zero_args = True
    elif kwargs:
        if 'force_test_define_by_run' in kwargs:
            force_test_define_by_run = kwargs['force_test_define_by_run']
        if 'minimize_cache_size' in kwargs:
            minimize_cache_size = kwargs['minimize_cache_size']
        if 'verbosity_level' in kwargs:
            verbosity_level = kwargs['verbosity_level']
        if 'enable_double_backprop' in kwargs:
            enable_double_backprop = kwargs['enable_double_backprop']

    def wrap(func):
        if False:
            while True:
                i = 10

        def wrapped_func(*inner_args, **inner_kwargs):
            if False:
                print('Hello World!')
            if not chainer.config.use_static_graph:
                return func(*inner_args, **inner_kwargs)
            if verbosity_level >= 2:
                print('Calling static chain...')
            chain = inner_args[0]
            chain_args = inner_args[1:]
            if chainer.config.train is False and force_test_define_by_run:
                return func(*inner_args, **inner_kwargs)
            (chain_args_flat, in_unflatten_inds, __) = _flatten_args(chain_args)
            flat_vars = []
            for x in chain_args_flat:
                if not isinstance(x, chainer.Variable):
                    flat_vars.append(chainer.Variable(x))
                else:
                    flat_vars.append(x)
            flat_vars = tuple(flat_vars)
            if not hasattr(chain, 'schedule_manager'):
                chain.schedule_manager = ScheduleManager(minimize_cache_size=minimize_cache_size, verbosity_level=verbosity_level)
            schedule_manager = chain.schedule_manager
            edb = enable_double_backprop
            chain.static_schedule = schedule_manager.get_schedule(flat_vars, enable_double_backprop=edb)
            if verbosity_level >= 2:
                print('Current schedule manager info: ', schedule_manager)
            if not chain.static_schedule.is_empty():
                if verbosity_level >= 2:
                    print('This is the 2nd or greater iteration. Calling the existing static schedule...')
                    chain.static_schedule.debug_print_ref_counts()
                out_vars_flat = chain.static_schedule.apply(flat_vars)
                out_vars = _unflatten_args(out_vars_flat, chain._out_vars_unflatten_inds)
            else:
                assert isinstance(chain, chainer.Chain)
                if verbosity_level >= 2:
                    print('This is the first iteration. Calling the define-by-run code.: ', func)
                if chainer.config.schedule_func is not None:
                    raise RuntimeError('Not allowed to nest static chains: ', chain)
                new_args = []
                new_args.append(chain)
                new_flat_vars = []
                for var in flat_vars:
                    new_flat_vars.append(chainer.Variable(var.data))
                unflat_in_args = _unflatten_args_as_list(new_flat_vars, in_unflatten_inds)
                for item in unflat_in_args:
                    new_args.append(item)
                inner_args = tuple(new_args)
                with chainer.using_config('schedule_func', chain.static_schedule):
                    out_vars = func(*inner_args, **inner_kwargs)
                (out_vars_flat_dbr, chain._out_vars_unflatten_inds, __) = _flatten_args(out_vars)
                sched_out_vars = list(out_vars_flat_dbr)
                chain.static_schedule.set_out_variables(sched_out_vars)
                chain.static_schedule.build_schedule(chain, new_flat_vars)
                out_vars_flat = chain.static_schedule.apply(flat_vars)
                out_vars = _unflatten_args(out_vars_flat, chain._out_vars_unflatten_inds)
                if verbosity_level >= 2:
                    print('Returing from 1st call of the static chain.')
            return out_vars
        return wrapped_func
    if zero_args:
        return wrap(callable_arg)
    else:
        return wrap

def _flatten_args(xs):
    if False:
        while True:
            i = 10
    'Flatten the input into a tuple of variables.\n\n    In the typical case, `xs` is a tuple or list of objects where each\n    object is either a variable, list, or tuple. In the case where it is\n    a list of tuple, the objects in the list or tuple could also be either\n    a variable, list or tuple. Although the non-list and non-tuple items\n    are typically an instance of variable, any object other than list or\n    tuple is allowed.\n\n    This function simply flattens the hierarchical lists/tuples so that all\n    objects that are deeply contained in `xs` that are non-list and non-tuple\n    will be returned in a single tuple.\n\n    Args:\n        xs:\n\n    Returns:\n        The flattened tuple, allong with the indecies and count so that the\n        items can be unflattened later (i.e., by calling `_unflatten_args()`.\n\n    fixme: does not work if xs is a variable only.\n    '
    inds = []
    ys = []
    i = 0
    if not isinstance(xs, (list, tuple)):
        inds.append(('s',))
        return ((xs,), inds, 0)
    for x in xs:
        if isinstance(x, (list, tuple)):
            (x, sub_inds, total) = _flatten_args(x)
            inds.append(('i', i, i + total, sub_inds))
            i += total
        else:
            x = [x]
            inds.append(('f', i))
            i += 1
        ys.extend([y for y in x])
    return (tuple(ys), inds, i)

def _unflatten_args(xs, inds):
    if False:
        for i in range(10):
            print('nop')
    ys = []
    for ind in inds:
        code = ind[0]
        if code == 's':
            return xs[0]
        elif code == 'i':
            (i_start, i_end, sub_inds) = ind[1:]
            y = _unflatten_args(xs[i_start:i_end], sub_inds)
        else:
            i = ind[1]
            y = xs[i]
        ys.append(y)
    return tuple(ys)

def _unflatten_args_as_list(xs, inds):
    if False:
        while True:
            i = 10
    ys = []
    for ind in inds:
        code = ind[0]
        if code == 's':
            return xs[0]
        elif code == 'i':
            (i_start, i_end, sub_inds) = ind[1:]
            y = _unflatten_args(xs[i_start:i_end], sub_inds)
        else:
            i = ind[1]
            y = xs[i]
        ys.append(y)
    return ys