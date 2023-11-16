import inspect
import chainer

def static_code(*dec_args, **dec_kwargs):
    if False:
        while True:
            i = 10
    'Decorator to mark a function for inclusion in the static schedule.\n\n    This decorator is used to mark a function or method to be included\n    in a static schedule. There are multiple types of static schedules, such\n    as "forward pass schedule", "backward pass schedule", "double backward\n    pass schedule" etc.. The type of schedule that the decorated function\'s\n    code is added to will depend on the context in which this decorator\n    is used. For example, the decorated code will be added to the\n    "forward pass schedule" if it is called while executing the define-by-\n    run code of a static subgraph. To inform the framework that a particular\n    portion of define-by-run code corresponds to a static subgraph, the\n    code should be placed inside the `__call__()` method of a chain and\n    then apply the `@static_graph` decorator to the `__call__()` method.\n    We will refer to such a chain as a "static chain."\n    This will cause any functions\n    decorated with `static_code` that are called while inside of `__call__()`\n    to be included in the forward pass static\n    schedule in the same order in which they were executed in the\n    define-by-run code.\n\n    Likewise, for any `FunctionNode` instances that are called inside\n    a static chain, any code that is run while inside the `backward()`\n    method that calls a function using this decorator will be added to\n    the corresponding "backward pass schedule."\n\n    Usage:\n\n    This decorator should be applied to any code called from a static chain\n    that needs to run each\n    iteration. This should only include the code that performs\n    the actual forward and/or backward computations and not include code\n    for initializing parameters, checking types, etc..\n\n    As long as a chain is marked as static, the framework\n    will automatically wrap any `FunctionNode` instances so that the\n    code inside their `forward()` and `backward()` methods is added to\n    the corresponding forward and backward static schedules, respectively.\n    As a result, any built-in Chainer function and\n    link calls will be automatically included in the static schedule.\n\n    However, there are two cases where the user will need to use this\n    decorator:\n\n    1. Code with side effects that is called from a static chain\'s define-by-\n    run code must be placed in a function decorated with `@static_code`.\n\n    2. Any user-defined links that contain code other chain Chainer\n    function calls that must run every iteration must place such code\n    in a function decorated with `@static_graph`.\n\n\n    This decorator can be applied to either a function or a method (usually\n    of a `FunctionNode`). There are no required arguments, and so a user can\n    apply it to "side effect" code to cause an operation to be executed each\n    iteration. The more usual use case is where the core framework code\n    will apply it to the all of (and only) the functions\n    that actually perform the computations needed to compute the forward\n    and backward passes.\n\n    The simplest usage is when we would like to force a particular\n    user-defined function to run each iteration. For example, such a function\n    might increment a counter, check conditions, and possibly print\n    information to the console. In this use, it is only required to add\n    this decorator to the function definition and then call it during\n    the first iteration from the context of the static chain\'s\n    `__call__()` method.\n\n    Passing and returing arrays:\n\n    If the function needs an array as an input argument that was\n    used elsewhere in the static schedule, it must appear as an\n    item in list of arrays that is supplied in the `inputs` keyword\n    argument. An example would be the typical case where one layer\n    in the network produces output activations `y` which are then\n    used as the input of the next layer. If the corresponding\n    `FunctionNode` instances wrap their computaions using this decorator,\n    this will result in multiple functions that operate on `y`.\n    The following constraints specify how\n    such arrays should be passed into and returned from a function\n    that uses this decorator.\n\n    If the function will return results in one or more arrays, there are\n    two options:\n\n    1. Write the results in-place into preallocated arrays that are\n    supplied in a list in the `outputs` keyword argument.\n\n    2. Dynamically allocate the result array(s) inside the function\n    and return them inside a tuple.\n\n    When two schedule functions\n    "func_A" and "func_B" operate on the same array `x`,\n    `x` must explicitly appear as an input argument and/or output\n    of both functions. For\n    example, it would be an error to have schedule function "func_A"\n    return a dynamically allocated array `x` and then have schedule\n    function "func_B" later\n    read from `x` without it appearing in "func_B"\'s `inputs` list.\n    Note that this would work during the first iteration, but during\n    the next iteration when "func_A" is called, it would allocate and\n    return a new array for `x` leading to "func_B" reading from the\n    stale reference to `x` from the previous iteration. This\n    usage is allowed in some special cases by the framework code, but\n    is not allowed for user-defined functions.\n\n    Performance notes:\n\n    The function should return any output arrays in-place\n    into pre-allocated arrays (1. above) when possible since this this allows\n    the scheduler to make tradeoffs\n    between computation efficiency and memory usage.\n    For example, this allows the use of a\n    completely static array allocations (no allocations after the first\n    iteration), if desired. If memory reduction is needed, the\n    scheduler may choose to delete the arrays in `inputs` once they are no\n    longer\n    needed in an iteration and then reallocate them again in the next\n    iteration just before the function is called. Note that\n    completely static array allocations are not possible if\n    any of the schedule functions return a tuple of dynamically allocated\n    arrays, as the existing chainer functions do.\n\n    The following optional arguments apply to the wrapped function or method.\n\n    Args (of this decorater):\n        func_name (str): An optional descriptive name that will be associated\n            with this function in the static schedule. It is intended\n            for debugging purposes.\n\n    Args (of the wrapped fuction):\n        inputs (list of ndarray): An optional keyword argument that\n            supplies all arrays needed as input by the function. If the\n            function needs an array that is used by another function\n            in the static schedule, it must appear in this list.\n        outputs (list of ndarray): An optional keyword argument that\n            supplies all output arrays of this function. These arrays\n            must already have been initialized to the correct shapes\n            and dtypes before the function is called. The function\n            must write its results in-place into these arrays. Any\n            output arrays that may be used inside another schedule\n            function must appear in this list.\n\n    Returns:\n        None or a tuple of ndarray: If the function dynamically\n        allocates its output arrays, they must be returned in a tuple\n        of arrays.\n\n    '
    func_name = None
    zero_args = False
    if len(dec_args) == 1 and (not dec_kwargs) and callable(dec_args[0]):
        callable_arg = dec_args[0]
        zero_args = True
    elif dec_kwargs:
        if 'func_name' in dec_kwargs:
            func_name = dec_kwargs['func_name']

    def wrap(func):
        if False:
            return 10

        def wrapped_func(*args, **kwargs):
            if False:
                return 10
            schedule_function = chainer.config.schedule_func
            if schedule_function is not None:
                assert chainer.config.use_static_graph
                ret = schedule_function.append_function(func, args, kwargs, func_name=func_name)
                if args:
                    instance = args[0]
                    if inspect.isclass(instance):
                        instance.schedule_func = schedule_function
            else:
                ret = func(*args, **kwargs)
            return ret
        return wrapped_func
    if zero_args:
        return wrap(callable_arg)
    else:
        return wrap

def static_forward_optimizations(func, inputs):
    if False:
        while True:
            i = 10
    'Perform checks needed for creation of a static schedule.\n\n    Check if `func` supports static graph optimizations. If not,\n    automatically wrap it to be compatible.\n\n    This function is called from the `FunctionNode` apply() method\n    in place of the original `func.forward(inputs)` call if\n    `chainer.config.schedule_func` is not None.\n\n    Args:\n        func (instance of FunctionNode):\n        inputs (tuple of ndarray): input arrays to `func`\n\n    Returns:\n        (tuple of ndarray): The outputs of the function.\n    '
    schedule_function = chainer.config.schedule_func
    if not func._supports_static_optimizations:
        if schedule_function.verbosity_level >= 2:
            print('Adding automatic static graph support to function: ', func)

        @static_code(func_name=str(func))
        def generic_static_forward(func, inputs):
            if False:
                return 10
            'Auto-wrap the supplied function.\n\n            func (instance of FunctionNode): The function to include in\n                the static schedule.\n            inputs (list of input arrays): The input arguments to `func`.\n\n            Returns: a tuple of output arrays.\n\n            '
            in_data = tuple(inputs)
            ret = func.forward(in_data)
            return ret
        return generic_static_forward(func, inputs=list(inputs))
    return func.forward(inputs)