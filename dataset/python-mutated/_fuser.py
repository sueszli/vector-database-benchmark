import contextlib
from typing import List, Tuple
import torch

@contextlib.contextmanager
def optimized_execution(should_optimize):
    if False:
        i = 10
        return i + 15
    "Context manager that controls whether the JIT's executor will run optimizations before executing a function."
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)

@contextlib.contextmanager
def fuser(name):
    if False:
        for i in range(10):
            print('nop')
    'Context manager that facilitates switching between backend fusers.\n\n    Valid names:\n    * ``fuser0`` - enables only legacy fuser\n    * ``fuser1`` - enables only NNC\n    * ``fuser2`` - enables only nvFuser\n    * ``fuser3`` - enables oneDNN Graph\n    '
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    old_llga_state = torch._C._jit_llga_enabled()
    if name == 'fuser0':
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == 'fuser1':
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == 'fuser2':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_llga_enabled(False)
    elif name == 'fuser3':
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(True)
    elif name == 'none':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    else:
        raise Exception(f'unrecognized fuser option (name: {name})')
    try:
        yield
    finally:
        if name in ['fuser1', 'fuser3']:
            torch._C._jit_set_profiling_executor(old_profiling_executor)
            torch._C._get_graph_executor_optimize(old_profiling_mode)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)
        torch._C._jit_set_llga_enabled(old_llga_state)
last_executed_optimized_graph = torch._C._last_executed_optimized_graph

def _get_differentiable_graph_node(node, diff_node):
    if False:
        return 10
    if node.kind() == 'prim::DifferentiableGraph':
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)

def _graph_for(self, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return _script_method_graph_for(self, self, *args, **kwargs)

def _script_method_graph_for(self, parent, *args, **kwargs):
    if False:
        return 10
    try:
        dbs = parent.get_debug_state()
        eps = list(dbs.execution_plans.values())
        assert len(eps) == 1
        graph = eps[0].graph.copy()
        fw_states = eps[0].code.differentiable_op_executor_states()
        diff_nodes: List[torch._C.Node] = []
        for n in graph.nodes():
            _get_differentiable_graph_node(n, diff_nodes)
        assert len(fw_states) == len(diff_nodes)
        for (n, state) in zip(diff_nodes, fw_states):
            fw_execution_plans = list(state.execution_plans.values())
            if len(fw_execution_plans) == 1:
                n.g_('Subgraph', fw_execution_plans[0].graph)
        return graph
    except Exception:
        self(*args, **kwargs)
        return last_executed_optimized_graph()

def set_fusion_strategy(strategy: List[Tuple[str, int]]):
    if False:
        for i in range(10):
            print('nop')
    'Set the type and number of specializations that can occur during fusion.\n\n    Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC"\n    and depth is an integer.\n\n    Behavior - static vs dynamic:\n        In STATIC fusion, fused ops are compiled to have fixed input shapes. The shape is determined\n        based on some initial profiling runs.\n        In DYNAMIC fusion, fused ops are compiled to have variable input shapes, so that multiple\n        shapes are possible.\n\n    In both cases, we also recompile on new striding behavior, device, or dtype.\n\n    Behavior - fallback functions & depth:\n        When an input doesn\'t match the format required by the specialized compiled op, it will run\n        a fallback function. Fallback functions are recursively be compiled and specialized based\n        on the observed tensor shapes. Since compilation can be slow, the "depth" parameter is provided to\n        limit the number of specializations that can be compiled, before giving up on recompiling and\n        falling back to a completely un-fused, un-specialized implementation.\n\n    The list of (type, depth) pairs controls the type of specializations and the number of\n    specializations. For example: [("STATIC", 2), ("DYNAMIC", 2)] indicates that the first\n    two specializations will use static fusions, the following two specializations will use\n    dynamic fusion, and any inputs that satisfy none of the 4 options will run an\n    unfused implementation.\n\n    NB: in the future, if more as more fusion backends are added there may be more granular\n    apis for specific fusers.\n    '
    return torch._C._jit_set_fusion_strategy(strategy)