import gc
import torch
from torch.utils import _pytree
from ._utils import _dummy_type
if not hasattr(torch._C, '_CudaStreamBase'):
    torch._C.__dict__['_CUDAGraph'] = _dummy_type('_CUDAGraph')
    torch._C.__dict__['_graph_pool_handle'] = _dummy_type('_graph_pool_handle')
    torch._C.__dict__['_cuda_isCurrentStreamCapturing'] = _dummy_type('_cuda_isCurrentStreamCapturing')
from torch._C import _cuda_isCurrentStreamCapturing, _CUDAGraph, _graph_pool_handle

def is_current_stream_capturing():
    if False:
        i = 10
        return i + 15
    'Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise.\n\n    If a CUDA context does not exist on the current device, returns False without initializing the context.\n    '
    return _cuda_isCurrentStreamCapturing()

def graph_pool_handle():
    if False:
        while True:
            i = 10
    'Return an opaque token representing the id of a graph memory pool.\n\n    See :ref:`Graph memory management<graph-memory-management>`.\n\n    .. warning::\n        This API is in beta and may change in future releases.\n    '
    return _graph_pool_handle()

class CUDAGraph(torch._C._CUDAGraph):
    """Wrapper around a CUDA graph.

    .. warning::
        This API is in beta and may change in future releases.
    """

    def __new__(cls):
        if False:
            return 10
        return super().__new__(cls)

    def capture_begin(self, pool=None, capture_error_mode='global'):
        if False:
            i = 10
            return i + 15
        'Begin capturing CUDA work on the current stream.\n\n        Typically, you shouldn\'t call ``capture_begin`` yourself.\n        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,\n        which call ``capture_begin`` internally.\n\n        Arguments:\n            pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or\n                :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory\n                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.\n            capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.\n                Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,\n                may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for\n                actions in the current thread, and "relaxed" will not error on these actions. Do NOT change this setting\n                unless you\'re familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_\n        '
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)

    def capture_end(self):
        if False:
            while True:
                i = 10
        "End CUDA graph capture on the current stream.\n\n        After ``capture_end``, ``replay`` may be called on this instance.\n\n        Typically, you shouldn't call ``capture_end`` yourself.\n        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,\n        which call ``capture_end`` internally.\n        "
        super().capture_end()

    def replay(self):
        if False:
            while True:
                i = 10
        'Replay the CUDA work captured by this graph.'
        super().replay()

    def reset(self):
        if False:
            print('Hello World!')
        'Delete the graph currently held by this instance.'
        super().reset()

    def pool(self):
        if False:
            return 10
        "Return an opaque token representing the id of this graph's memory pool.\n\n        This id can optionally be passed to another graph's ``capture_begin``,\n        which hints the other graph may share the same memory pool.\n        "
        return super().pool()

    def enable_debug_mode(self):
        if False:
            for i in range(10):
                print('nop')
        'Enable debugging mode for CUDAGraph.debug_dump.'
        return super().enable_debug_mode()

    def debug_dump(self, debug_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arguments:\n            debug_path (required): Path to dump the graph to.\n\n        Calls a debugging function to dump the graph if the debugging is\n        enabled via CUDAGraph.enable_debug_mode()\n        '
        return super().debug_dump(debug_path)

class graph:
    """Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
        capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
            Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
            may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
            actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
            unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.

    .. _cudaStreamCaptureMode:
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
    """
    default_capture_stream = None

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode: str='global'):
        if False:
            for i in range(10):
                print('nop')
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()
        self.pool = () if pool is None else (pool,)
        self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
        assert self.capture_stream is not None
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(*self.pool, capture_error_mode=self.capture_error_mode)

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            return 10
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)

def make_graphed_callables(callables, sample_args, num_warmup_iters=3, allow_unused_input=False):
    if False:
        while True:
            i = 10
    "Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\\ s) and returns graphed versions.\n\n    Each graphed callable's forward pass runs its source callable's\n    forward CUDA work as a CUDA graph inside a single autograd node.\n\n    The graphed callable's forward pass also appends\n    a backward node to the autograd graph. During backward, this node runs the\n    callable's backward work as a CUDA graph.\n\n    Therefore, each graphed callable should be a drop-in replacement for its source callable\n    in an autograd-enabled training loop.\n\n    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.\n\n    If you pass a tuple of several callables, their captures will use the same memory pool.\n    See :ref:`Graph memory management<graph-memory-management>` for when this is appropriate.\n\n    Arguments:\n        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.\n            See :ref:`Graph memory management<graph-memory-management>` for when passing a tuple of callables\n            is appropriate.  If you pass a tuple of callables, their order in the tuple must be the same order\n            they'll run in the live workload.\n        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.\n            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.\n            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.\n        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs\n            11 iterations for warm up. Default: ``3``.\n        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs\n            (and therefore their grad is always zero) is an error. Defaults to False.\n\n    .. note::\n        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state\n        that's expected for the corresponding real input in the training loop.\n\n    .. warning::\n        This API is in beta and may change in future releases.\n\n    .. warning::\n        ``sample_args`` for each callable must contain only Tensors. Other types are not allowed.\n\n    .. warning::\n        Returned callables do not support higher order differentiation (e.g., double backward).\n\n    .. warning::\n        In any :class:`~torch.nn.Module` passed to :func:`~make_graphed_callables`, only parameters\n        may be trainable. Buffers must have ``requires_grad=False``.\n\n    .. warning::\n        After you pass a :class:`torch.nn.Module` through :func:`~make_graphed_callables`,\n        you may not add or remove any of that Module's parameters or buffers.\n\n    .. warning::\n        :class:`torch.nn.Module`\\s passed to :func:`~torch.cuda.make_graphed_callables` must not have module hooks\n        registered on them at the time they are passed. However, registering hooks on modules *after* passing them\n        through :func:`~torch.cuda.make_graphed_callables` is allowed.\n\n    .. warning::\n        When running a graphed callable, you must pass its arguments in the same order and format\n        they appeared in that callable's ``sample_args``.\n\n    .. warning::\n        The automatic mixed precision is supported in :func:`~torch.cuda.make_graphed_callables` only with disabled\n        caching. The context manager `torch.cuda.amp.autocast()` must have `cache_enabled=False`.\n    "
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError('make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`.')
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
    flatten_sample_args = []
    for (c, args) in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and (len(c._forward_pre_hooks) == 0), 'Modules must not have hooks registered at the time they are passed. However, registering hooks ' + 'on modules after passing them through make_graphed_callables is allowed.'
            assert all((b.requires_grad is False for b in c.buffers())), 'In any :class:`~torch.nn.Module` passed to ' + ':func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have ' + '``requires_grad=False``.'
        flatten_arg = _pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        assert all((isinstance(arg, torch.Tensor) for arg in flatten_arg)), 'In the beta API, sample_args ' + 'for each callable must contain only Tensors. Other types are not allowed.'
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables]
    per_callable_static_input_surfaces = [flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))]
    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    mempool = graph_pool_handle()
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for (func, args, static_input_surface) in zip(callables, sample_args, per_callable_static_input_surfaces):
            for _ in range(num_warmup_iters):
                outputs = _pytree.tree_leaves(func(*args))
                grad_inputs = torch.autograd.grad(outputs=tuple((o for o in outputs if o.requires_grad)), inputs=tuple((i for i in static_input_surface if i.requires_grad)), grad_outputs=tuple((torch.empty_like(o) for o in outputs if o.requires_grad)), only_inputs=True, allow_unused=allow_unused_input)
            del outputs, grad_inputs
    torch.cuda.synchronize()
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for (func, args, fwd_graph) in zip(callables, sample_args, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args)
        (flatten_outputs, spec) = _pytree.tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for (static_input_surface, static_outputs, bwd_graph, module_params) in zip(reversed(per_callable_static_input_surfaces), reversed(per_callable_static_outputs), reversed(bwd_graphs), reversed(per_callable_module_params)):
        static_grad_outputs = tuple((torch.empty_like(o) if o.requires_grad else None for o in static_outputs))
        with torch.cuda.graph(bwd_graph, pool=mempool):
            grad_inputs = torch.autograd.grad(outputs=tuple((o for o in static_outputs if o.requires_grad)), inputs=tuple((i for i in static_input_surface if i.requires_grad)), grad_outputs=tuple((o for o in static_grad_outputs if o is not None)), only_inputs=True, allow_unused=allow_unused_input)
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)
        static_grad_inputs = tuple(static_grad_inputs)
        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)
    per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
    per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))

    def make_graphed_autograd_function(fwd_graph, bwd_graph, module_params, len_user_args, output_unflatten_spec, static_input_surface, static_outputs, static_grad_outputs, static_grad_inputs):
        if False:
            print('Hello World!')

        class Graphed(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *inputs):
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple((o.detach() for o in static_outputs))

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                if False:
                    while True:
                        i = 10
                assert len(grads) == len(static_grad_outputs)
                for (g, grad) in zip(static_grad_outputs, grads):
                    if g is not None:
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()
                assert isinstance(static_grad_inputs, tuple)
                return tuple((b.detach() if b is not None else b for b in static_grad_inputs))

        def functionalized(*user_args):
            if False:
                while True:
                    i = 10
            flatten_user_args = _pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*tuple(flatten_user_args) + module_params)
            return _pytree.tree_unflatten(out, output_unflatten_spec)
        return functionalized
    ret = []
    for (i, func) in enumerate(callables):
        graphed = make_graphed_autograd_function(fwd_graphs[i], bwd_graphs[i], per_callable_module_params[i], per_callable_len_user_args[i], per_callable_output_unflatten_spec[i], per_callable_static_input_surfaces[i], per_callable_static_outputs[i], per_callable_static_grad_outputs[i], per_callable_static_grad_inputs[i])
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                if False:
                    return 10

                def new_fwd(*user_args):
                    if False:
                        i = 10
                        return i + 15
                    if func.training == graph_training_state:
                        return graphed(*user_args)
                    else:
                        return orig_fwd(*user_args)
                return new_fwd
            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)
            ret.append(func)
        else:
            ret.append(graphed)
    if just_one_callable:
        return ret[0]
    return tuple(ret)