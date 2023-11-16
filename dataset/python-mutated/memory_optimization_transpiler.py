import logging

def memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=True):
    if False:
        return 10
    '\n        :api_attr: Static Graph\n\n    This API is deprecated since 1.6. Please do not use it. The better\n    memory optimization strategies are enabled by default.\n    '
    logging.warn('Caution! paddle.distributed.transpiler.memory_optimize() is deprecated and not maintained any more, since it is not stable!\nThis API would not take any memory optimizations on your Program now, since we have provided default strategies for you.\nThe newest and stable memory optimization strategies (they are all enabled by default) are as follows:\n 1. Garbage collection strategy, which is enabled by exporting environment variable FLAGS_eager_delete_tensor_gb=0 (0 is the default value).\n 2. Inplace strategy, which is enabled by setting build_strategy.enable_inplace=True (True is the default value) when using CompiledProgram or ParallelExecutor.\n')

def release_memory(input_program, skip_opt_set=None):
    if False:
        i = 10
        return i + 15
    '\n        :api_attr: Static Graph\n\n    This API is deprecated since 1.6. Please do not use it. The better\n    memory optimization strategies are enabled by default.\n    '
    logging.warn('paddle.distributed.transpiler.release_memory() is deprecated, it would not take any memory release on your program')