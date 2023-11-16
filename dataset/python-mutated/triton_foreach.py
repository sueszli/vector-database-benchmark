import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sympy import Integer
from .. import metrics
from ..scheduler import SchedulerNode
from ..utils import ceildiv, Placeholder
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_to_meta

@dataclass
class PartitionState:
    partitions: List[List[Tuple[List[SchedulerNode], Tuple[Integer, ...], Integer, Integer]]]
    cur_partition: List[Tuple[List[SchedulerNode], Tuple[Integer, ...], Integer, Integer]]
    cur_count: int

    def finalize(self):
        if False:
            print('Hello World!')
        if self.cur_partition:
            self.partitions.append(self.cur_partition)

class ForeachKernel(Kernel):
    MAX_NUM_ARGS = 250

    @staticmethod
    def _update_partition(partition_state, node_rw_count, node_info):
        if False:
            while True:
                i = 10
        if partition_state.cur_count + node_rw_count > ForeachKernel.MAX_NUM_ARGS:
            partition_state.partitions.append(partition_state.cur_partition)
            partition_state.cur_partition = [node_info]
            partition_state.cur_count = node_rw_count
        else:
            partition_state.cur_count += node_rw_count
            partition_state.cur_partition.append(node_info)

    @staticmethod
    def horizontal_partition(subkernel_nodes, triton_scheduling):
        if False:
            print('Hello World!')
        'Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)\n        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args\n        (read/writes) and to have the same 2D or 1D blocking strategy.'
        assert len(subkernel_nodes) >= 1
        partition_state_1d = PartitionState([], [], 0)
        yelem_to_partition_state_2d: Dict[Integer, PartitionState] = defaultdict(lambda : PartitionState([], [], 0))
        for node in subkernel_nodes:
            fused_nodes = node.get_nodes()
            (_, (numel, rnumel)) = max(fused_nodes, key=lambda x: int(x.is_reduction())).group
            tiled_groups = triton_scheduling.select_tiling(fused_nodes, numel, rnumel)
            node_info = (fused_nodes, tiled_groups, numel, rnumel)
            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)
            if tiled_groups[1] == 1:
                ForeachKernel._update_partition(partition_state_1d, read_write_count, node_info)
            else:
                y_elem = tiled_groups[0]
                partition_state_2d = yelem_to_partition_state_2d[y_elem]
                ForeachKernel._update_partition(partition_state_2d, read_write_count, node_info)
        partition_state_1d.finalize()
        all_partitions = partition_state_1d.partitions
        for partition_state_2d in yelem_to_partition_state_2d.values():
            partition_state_2d.finalize()
            all_partitions.extend(partition_state_2d.partitions)
        return all_partitions

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.blocking_2d = False
        self.block_size_1d = 1024
        self.block_size_2d = 32
        self.num_warps = 8
        self.sub_kernels = []
        self.iter_vars_count = itertools.count()
        self.x_block_count = 0
        self.y_block_count = 0

    def get_block_size(self):
        if False:
            return 10
        if self.blocking_2d:
            return self.block_size_2d
        else:
            return self.block_size_1d

    @staticmethod
    def codegen_pid_offsets(code, block_count, lower_bound, prefix):
        if False:
            i = 10
            return i + 15
        if block_count == 0:
            code.splice(f'{prefix}pid_offset = {prefix}pid')
        else:
            code.splice(f'{prefix}pid_offset = {prefix}pid - {lower_bound}')

    def codegen_pid_range(self, code, x_elems):
        if False:
            while True:
                i = 10
        num_x_blocks = ceildiv(x_elems, self.get_block_size())
        upper_bound_x_pid = self.x_block_count + num_x_blocks
        lower_bound_x_pid = self.x_block_count
        if self.x_block_count == 0:
            cond = 'if'
        else:
            cond = 'elif'
        x_pid_bounds_check = f'xpid >= {lower_bound_x_pid} and xpid < {upper_bound_x_pid}'
        code.splice(f'{cond} {x_pid_bounds_check}:')
        with code.indent():
            ForeachKernel.codegen_pid_offsets(code, num_x_blocks, lower_bound_x_pid, 'x')
            self.x_block_count += num_x_blocks

    def create_sub_kernel(self, *groups, index_dtype, mutations, reduction_hint):
        if False:
            return 10
        sub_kernel = TritonKernel(*groups, index_dtype=index_dtype, mutations=mutations, pid_cache={'tl.program_id(0)': 'xpid_offset', 'tl.program_id(1)': 'ypid'}, reduction_hint=reduction_hint)
        if self.blocking_2d:
            assert len(groups) == 3
        self.blocking_2d |= groups[1] != 1 and len(groups) == 3
        metrics.generated_kernel_count -= 1
        sub_kernel.args = self.args
        sub_kernel.iter_vars_count = self.iter_vars_count
        sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
        self.sub_kernels.append(sub_kernel)
        return sub_kernel

    def jit_line(self):
        if False:
            i = 10
            return i + 15
        can_use_32bit = all((k.index_dtype == 'tl.int32' for k in self.sub_kernels))
        size_dtype = 'tl.int32' if can_use_32bit else 'tl.int64'
        (_, _, signature) = self.args.python_argdefs()
        triton_meta = {'signature': signature_to_meta(signature, size_dtype=size_dtype), 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': {}}
        triton_meta['configs'] = [config_of(signature)]
        inductor_meta = {'kernel_name': str(Placeholder.DESCRIPTIVE_NAME)}
        return f'@foreach(num_warps={self.num_warps}, triton_meta={triton_meta!r}, inductor_meta={inductor_meta!r})\n' + '@triton.jit'

    def grid(self):
        if False:
            while True:
                i = 10
        return (self.x_block_count, ceildiv(int(self.sub_kernels[0].numels[0]), self.block_size_2d) if self.blocking_2d else 1, 1)

    def codegen_kernel(self, name=None):
        if False:
            return 10
        code = IndentedBuffer()
        code.splice('\n                import triton\n                import triton.language as tl\n                from torch._inductor.triton_heuristics import foreach\n                from torch._inductor.utils import instance_descriptor\n                from torch._inductor import triton_helpers\n            ')
        (argdefs, _, _) = self.args.python_argdefs()
        code.writeline(self.jit_line())
        code.writeline(f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):")
        with code.indent():
            code.splice('xpid = tl.program_id(0)')
            if self.blocking_2d:
                code.splice('ypid = tl.program_id(1)')
                code.splice(f'XBLOCK: tl.constexpr = {self.block_size_2d}')
                code.splice(f'YBLOCK: tl.constexpr = {self.block_size_2d}')
            else:
                code.splice(f'XBLOCK: tl.constexpr = {self.block_size_1d}')
            for sub_kernel in self.sub_kernels:
                assert len(sub_kernel.numels) <= 3
                numel_ind = 0 if not self.blocking_2d else 1
                self.codegen_pid_range(code, int(sub_kernel.numels[numel_ind]))
                with code.indent():
                    if self.blocking_2d:
                        code.splice(f'ynumel = {sub_kernel.numels[0]}')
                        code.splice(f'xnumel = {sub_kernel.numels[1]}')
                    else:
                        code.splice(f'xnumel = {sub_kernel.numels[0]}')
                    sub_kernel.codegen_body()
                    code.splice(sub_kernel.body)
            code.splice('else:')
            with code.indent():
                code.splice('pass')
        return code.getvalue()

    def call_kernel(self, code, name: str):
        if False:
            return 10
        (_, call_args, _) = self.args.python_argdefs()
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + '.item()'
        if V.graph.cpp_wrapper:
            V.graph.wrapper_code.generate_kernel_call(name, call_args, device_index=V.graph.scheduler.current_device.index, grid=self.grid())
        else:
            call_args_str = ', '.join(call_args)
            stream_name = code.write_get_raw_stream(V.graph.scheduler.current_device.index)
            code.writeline(f'{name}.run({call_args_str}, grid=({self.grid()}), stream={stream_name})')