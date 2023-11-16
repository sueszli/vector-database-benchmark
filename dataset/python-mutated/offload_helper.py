from paddle.framework import core
from paddle.utils import unique_name
from ..common import OP_ROLE_KEY, OpRole, is_optimizer_op, is_update_op
__all__ = []

class PlaceType:
    CPU = 0
    CUDA = 1
    CUDA_PINNED = 2
    XPU = 3

    @staticmethod
    def default_device():
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            return PlaceType.CUDA
        return PlaceType.CPU

    @staticmethod
    def default_pinned():
        if False:
            return 10
        if core.is_compiled_with_cuda():
            return PlaceType.CUDA_PINNED
        return PlaceType.CPU

class OffloadHelper:
    cpu_place_type = 0
    cuda_place_type = PlaceType.default_device()
    cuda_pinned_place_type = PlaceType.default_pinned()

    def __init__(self, mp_ring_id=None, dp_ring_id=None):
        if False:
            for i in range(10):
                print('nop')
        self.mp_ring_id = mp_ring_id
        self.dp_ring_id = dp_ring_id

    def _insert_cast_op(self, block, idx, src_name, dst_name):
        if False:
            return 10
        src_var = block.var(src_name)
        if not block.has_var(dst_name):
            block.create_var(name=dst_name, shape=src_var.shape, dtype=core.VarDesc.VarType.FP16, persistable=True)
        dst_var = block.var(dst_name)
        assert dst_var.dtype == core.VarDesc.VarType.FP16
        block._insert_op_without_sync(idx, type='cast', inputs={'X': src_var}, outputs={'Out': dst_var}, attrs={'in_dtype': src_var.dtype, 'out_dtype': dst_var.dtype, OP_ROLE_KEY: OpRole.Optimize})

    def _insert_broadcast_op(self, block, idx, param_name):
        if False:
            while True:
                i = 10
        rings = []
        if self.dp_ring_id is not None:
            rings.append(self.dp_ring_id)
        if self.mp_ring_id is not None:
            param = block.var(param_name)
            if not hasattr(param, 'is_distributed') or not param.is_distributed:
                rings.append(self.mp_ring_id)
        for ring in rings:
            block._insert_op_without_sync(idx, type='c_broadcast', inputs={'X': param_name}, outputs={'Out': param_name}, attrs={'ring_id': ring, 'root': 0, 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Forward})

    def _insert_memcpy_op(self, block, idx, src_name, dst_name, dst_place_type):
        if False:
            i = 10
            return i + 15
        src_var = block.var(src_name)
        dst_var = block.var(dst_name)
        block._insert_op_without_sync(idx, type='memcpy', inputs={'X': src_var}, outputs={'Out': dst_var}, attrs={'dst_place_type': dst_place_type, OP_ROLE_KEY: OpRole.Optimize})

    def _insert_fetch_op(self, block, idx, src_name, dst_name):
        if False:
            i = 10
            return i + 15
        self._insert_memcpy_op(block, idx, src_name, dst_name, OffloadHelper.cuda_place_type)

    def _insert_offload_op(self, block, idx, src_name, dst_name):
        if False:
            print('Hello World!')
        self._insert_memcpy_op(block, idx, src_name, dst_name, OffloadHelper.cuda_pinned_place_type)

    def _get_offload_var_name(self, name):
        if False:
            print('Hello World!')
        return unique_name.generate(name + '@offload')

    def _create_offload_var(self, var_name, offload_var_name, blocks):
        if False:
            print('Hello World!')
        for block in blocks:
            var = block.var(var_name)
            var.persistable = False
            offload_var = block.create_var(name=offload_var_name, shape=var.shape, dtype=var.dtype, persistable=True)

    def offload_fp32param(self, block, startup_block, offload=True):
        if False:
            while True:
                i = 10
        '\n        (p_fp16) = cast(p)\n        (p_fp16_recompute) = cast(p)\n        (pout,) = adam(p)\n        ===========================>\n        rename(p_fp16_recompute, p_fp16)\n\n        (p,) = prefetch(p@offload)\n        (pout,) = adam(p)\n        (p_fp16) = cast(p)\n        (p@offload) = memcpy(p)\n        '
        param_to_idx = {}
        param_to_fp16 = {}
        fp16_param_to_recompute = {}
        recompute_to_fp16 = {}

        def remove_param(input_name):
            if False:
                return 10
            param_to_idx.pop(input_name)
            if input_name in param_to_fp16:
                fp16_param = param_to_fp16.pop(input_name)
                if fp16_param in fp16_param_to_recompute:
                    recompute = fp16_param_to_recompute.pop(fp16_param)
                    recompute_to_fp16.pop(recompute)
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if is_update_op(op):
                param = op.desc.input('Param')[0]
                param_to_idx[param] = idx
        for (idx, op) in enumerate(block.ops):
            if is_optimizer_op(op):
                break
            if not offload and op.type == 'coalesce_tensor':
                continue
            for input_name in op.desc.input_arg_names():
                if input_name not in param_to_idx:
                    continue
                if op.type != 'cast':
                    remove_param(input_name)
                    continue
                output_name = op.output_arg_names[0]
                if 'cast_fp16' not in output_name:
                    remove_param(input_name)
                    continue
                if 'subprog' not in output_name:
                    assert output_name == input_name + '.cast_fp16'
                    assert input_name not in param_to_fp16, 'There must be only one cast op from fp32 param to fp16 param.'
                    param_to_fp16[input_name] = output_name
                else:
                    assert input_name in param_to_fp16, 'param must first be cast to fp16'
                    fp16_param = param_to_fp16[input_name]
                    fp16_param_to_recompute[fp16_param] = output_name
                    recompute_to_fp16[output_name] = fp16_param
        param_name_to_offload_name = {}
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if is_update_op(op):
                param = op.desc.input('Param')[0]
                if param not in param_to_idx:
                    continue
                offload_var_name = self._get_offload_var_name(param)
                param_name_to_offload_name[param] = offload_var_name
                if offload:
                    self._create_offload_var(param, offload_var_name, [block, startup_block])
                    self._insert_offload_op(block, idx + 1, param, offload_var_name)
                assert param in param_to_fp16
                fp16_param_name = param_to_fp16[param]
                fp16_param_var = block.var(fp16_param_name)
                fp16_param_var.persistable = True
                self._insert_cast_op(block, idx + 1, param, param_to_fp16[param])
                if offload:
                    self._insert_fetch_op(block, idx, offload_var_name, param)
                continue
            if op.type == 'cast':
                input_name = op.desc.input_arg_names()[0]
                if input_name in param_to_idx:
                    block._remove_op(idx, sync=False)
                    continue
            for input_name in op.desc.input_arg_names():
                if input_name in recompute_to_fp16:
                    op._rename_input(input_name, recompute_to_fp16[input_name])
            for output_name in op.desc.output_arg_names():
                if output_name in recompute_to_fp16:
                    op._rename_output(output_name, recompute_to_fp16[output_name])
        for name in recompute_to_fp16.keys():
            block._remove_var(name, sync=False)
        visited_vars = set()
        insert_idx = len(startup_block.ops)
        for (idx, op) in reversed(list(enumerate(startup_block.ops))):
            for out_name in op.output_arg_names:
                if out_name in visited_vars:
                    continue
                if out_name in param_name_to_offload_name:
                    var_name = out_name
                    if offload:
                        offload_var_name = param_name_to_offload_name[var_name]
                        self._insert_offload_op(startup_block, insert_idx, var_name, offload_var_name)
                    self._insert_cast_op(startup_block, insert_idx, var_name, param_to_fp16[var_name])
                    self._insert_broadcast_op(startup_block, insert_idx, var_name)
                visited_vars.add(out_name)
        block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def cast_fp32param_in_optimize(self, block, startup_block):
        if False:
            while True:
                i = 10
        '\n        (p_fp16) = cast(p)\n        (p_fp16_recompute) = cast(p)\n        (pout,) = adam(p)\n        ===========================>\n        rename(p_fp16_recompute, p_fp16)\n\n        (pout,) = adam(p)\n        (p_fp16) = cast(p)\n        '
        self.offload_fp32param(block, startup_block, offload=False)

    def offload(self, block, startup_block):
        if False:
            for i in range(10):
                print('nop')
        '\n        (m1, m2) = prefetch(m1@offload, m2@offload)\n        (m1out, m2out, pout) = adam(m1, m2, p)\n        (m1@offload, m2@offload) = memcpy(m1, m2)\n        '
        vars_name_to_offload_name = {}
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if not is_optimizer_op(op):
                break
            vars_name = []
            if op.type == 'adam' or op.type == 'adamw':
                vars_name.append(op.desc.input('Moment1')[0])
                vars_name.append(op.desc.input('Moment2')[0])
            elif op.type == 'momentum':
                pass
            elif op.type == 'lars':
                pass
            elif op.type == 'lamb':
                pass
            for var_name in vars_name:
                assert var_name not in vars_name_to_offload_name
                offload_var_name = self._get_offload_var_name(var_name)
                vars_name_to_offload_name[var_name] = offload_var_name
                self._create_offload_var(var_name, offload_var_name, [block, startup_block])
            for var_name in vars_name:
                offload_var_name = vars_name_to_offload_name[var_name]
                self._insert_offload_op(block, idx + 1, var_name, offload_var_name)
            for var_name in vars_name:
                offload_var_name = vars_name_to_offload_name[var_name]
                self._insert_fetch_op(block, idx, offload_var_name, var_name)
        visited_vars = set()
        for (idx, op) in reversed(list(enumerate(startup_block.ops))):
            for out_name in op.output_arg_names:
                if out_name in visited_vars:
                    continue
                if out_name in vars_name_to_offload_name:
                    var_name = out_name
                    offload_var_name = vars_name_to_offload_name[var_name]
                    self._insert_offload_op(startup_block, idx + 1, var_name, offload_var_name)
                visited_vars.add(out_name)
        block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def opt_sharding_cast_fp32param(self, block, startup_block, params, offload=False):
        if False:
            return 10
        '\n        (p_fp16) = cast(p)\n        (p_fp16_recompute) = cast(p)\n        (pout,) = adam(p)\n        ===========================>\n        rename(p_fp16_recompute, p_fp16)\n\n        (pout,) = adam(p)\n        (p_fp16) = cast(p)\n        broadcast(p_fp16)\n        '
        global_params = set()
        local_params = set()
        param_to_fp16 = {}
        fp16_param_to_recompute = {}
        recompute_to_fp16 = {}

        def remove_param(input_name):
            if False:
                print('Hello World!')
            global_params.remove(input_name)
            if input_name in local_params:
                local_params.remove(input_name)
            if input_name in param_to_fp16:
                fp16_param = param_to_fp16.pop(input_name)
                if fp16_param in fp16_param_to_recompute:
                    recompute = fp16_param_to_recompute.pop(fp16_param)
                    recompute_to_fp16.pop(recompute)
        global_params = set(params)
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if is_update_op(op):
                param = op.desc.input('Param')[0]
                local_params.add(param)
        for (idx, op) in enumerate(block.ops):
            if is_optimizer_op(op):
                break
            if op.type == 'coalesce_tensor':
                continue
            for input_name in op.desc.input_arg_names():
                if input_name not in global_params:
                    continue
                if op.type != 'cast':
                    remove_param(input_name)
                    continue
                output_name = op.output_arg_names[0]
                if 'cast_fp16' not in output_name:
                    remove_param(input_name)
                    continue
                if 'subprog' not in output_name:
                    assert output_name == input_name + '.cast_fp16'
                    assert input_name not in param_to_fp16, 'There must be only one cast op from fp32 param to fp16 param.'
                    param_to_fp16[input_name] = output_name
                else:
                    assert input_name in param_to_fp16, 'param must first be cast to fp16'
                    fp16_param = param_to_fp16[input_name]
                    fp16_param_to_recompute[fp16_param] = output_name
                    recompute_to_fp16[output_name] = fp16_param
        param_name_to_offload_name = {}
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if is_update_op(op):
                param = op.desc.input('Param')[0]
                if param not in global_params:
                    continue
                offload_var_name = self._get_offload_var_name(param)
                param_name_to_offload_name[param] = offload_var_name
                if offload:
                    self._create_offload_var(param, offload_var_name, [block, startup_block])
                    self._insert_offload_op(block, idx + 1, param, offload_var_name)
                assert param in param_to_fp16
                fp16_param_name = param_to_fp16[param]
                fp16_param_var = block.var(fp16_param_name)
                fp16_param_var.persistable = True
                self._insert_cast_op(block, idx + 1, param, param_to_fp16[param])
                if offload:
                    self._insert_fetch_op(block, idx, offload_var_name, param)
                continue
            if op.type == 'cast':
                input_name = op.desc.input_arg_names()[0]
                if input_name in global_params:
                    block._remove_op(idx, sync=False)
                    continue
            for input_name in op.desc.input_arg_names():
                if input_name in recompute_to_fp16:
                    op._rename_input(input_name, recompute_to_fp16[input_name])
            for output_name in op.desc.output_arg_names():
                if output_name in recompute_to_fp16:
                    op._rename_output(output_name, recompute_to_fp16[output_name])
        for name in recompute_to_fp16.keys():
            block._remove_var(name, sync=False)
        for (idx, op) in enumerate(block.ops):
            if op.type not in ['coalesce_tensor', 'c_broadcast']:
                continue
            for input_name in op.desc.input_arg_names():
                if input_name in param_to_fp16:
                    op._rename_input(input_name, param_to_fp16[input_name])
            for output_name in op.desc.output_arg_names():
                if output_name in param_to_fp16:
                    op._rename_output(output_name, param_to_fp16[output_name])
        for param in global_params:
            assert param in param_to_fp16
            fp16_param_name = param_to_fp16[param]
            fp16_param_var = block.var(fp16_param_name)
            fp16_param_var.persistable = True
            if param not in local_params:
                block._remove_var(param, sync=False)
        visited_vars = set()
        insert_idx = len(startup_block.ops)
        for (idx, op) in reversed(list(enumerate(startup_block.ops))):
            for out_name in op.output_arg_names:
                if out_name in visited_vars:
                    continue
                if out_name in param_to_fp16:
                    var_name = out_name
                    if offload:
                        self._insert_offload_op(startup_block, idx + 1, var_name, param_name_to_offload_name[var_name])
                    self._insert_cast_op(startup_block, insert_idx, var_name, param_to_fp16[var_name])
                    self._insert_broadcast_op(startup_block, insert_idx, var_name)
                    if var_name not in local_params:
                        param = startup_block.var(out_name)
                        param.persistable = False
                visited_vars.add(out_name)
        block._sync_with_cpp()
        startup_block._sync_with_cpp()