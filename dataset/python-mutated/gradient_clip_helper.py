from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
__all__ = []

class GradientClipHelper:

    def __init__(self, mp_ring_id):
        if False:
            return 10
        self.mp_ring_id = mp_ring_id

    def _is_gradient_clip_op(self, op):
        if False:
            while True:
                i = 10
        return op.desc.has_attr('op_namescope') and op.desc.attr('op_namescope').startswith('/gradient_clip')

    def prune_gradient_clip(self, block, shard, ring_ids):
        if False:
            for i in range(10):
                print('nop')
        '\n        prune gradient_clip related ops for params that not belong to cur shard\n        prune: square, reduce_sum, elementwise_mul\n        keep: sum, sqrt, elementwise_max, elementwise_div\n        '
        deperated_vars = set()
        deperate_op_idx = set()
        reversed_x_paramname = []
        global_norm_sum_op_idx = -1
        for (idx, op) in enumerate(block.ops):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                global_norm_sum_op_idx = idx
                continue
            deperate_op = False
            for input_name in op.desc.input_arg_names():
                if input_name in deperated_vars:
                    deperate_op = True
                if '@MERGED' in input_name:
                    param_name = input_name.strip('@GRAD@MERGED')
                else:
                    param_name = input_name.strip('@GRAD')
                if shard.is_param(param_name) and (not shard.has_param(param_name)):
                    deperate_op = True
                elif shard.is_param(param_name):
                    reversed_x_paramname.append(param_name)
            if deperate_op:
                deperate_op_idx.add(idx)
                for output_name in op.desc.output_arg_names():
                    if output_name not in op.desc.input_arg_names():
                        deperated_vars.add(output_name)
        if not deperated_vars and global_norm_sum_op_idx == -1:
            return
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if not self._is_gradient_clip_op(op):
                continue
            if idx in deperate_op_idx:
                block._remove_op(idx, sync=False)
                continue
            if op.type == 'sum':
                reversed_inputs = []
                global_norm_sum_op_idx = idx
                for input_name in op.desc.input_arg_names():
                    if input_name not in deperated_vars:
                        reversed_inputs.append(input_name)
                op.desc.set_input('X', reversed_inputs)
                assert len(op.desc.output_arg_names()) == 1
                sum_res = op.desc.output_arg_names()[0]
                if len(reversed_inputs) == 0:
                    sum_var = block.var(sum_res)
                    namescope = op.attr('op_namescope')
                    block._remove_op(idx, sync=False)
                    op = block._insert_op_without_sync(idx, type='fill_constant', inputs={}, outputs={'Out': sum_res}, attrs={'shape': sum_var.shape, 'dtype': sum_var.dtype, 'value': 0.0, OP_ROLE_KEY: OpRole.Optimize})
                    op._set_attr('op_namescope', namescope)
                idx_offset = 1
                for ring_id in ring_ids:
                    if ring_id == -1:
                        continue
                    block._insert_op_without_sync(idx + idx_offset, type='c_allreduce_sum', inputs={'X': sum_res}, outputs={'Out': sum_res}, attrs={'ring_id': ring_id, 'op_namescope': '/gradient_clip_model_parallelism', 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Optimize})
                    idx_offset += 1
        to_check_param = set(reversed_x_paramname)
        should_check_param = set(shard.global_params).intersection({param for (param, worker_idx) in shard.global_param2device.items() if worker_idx == shard.worker_idx})
        assert to_check_param == should_check_param, 'amp check_finite_and_unscale         checking miss [{}] and got unexpected [{}]'.format(should_check_param - to_check_param, to_check_param - should_check_param)
        for var_name in deperated_vars:
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()
        return

    def sync_global_norm(self, block, ring_ids, mp_rank):
        if False:
            print('Hello World!')
        '\n        prune gradient_clip related ops for params that not belong to cur shard\n        prune: square, reduce_sum, elementwise_mul\n        keep: sum, sqrt, elementwise_max, elementwise_div\n        '
        is_clip_grad_by_global_norm = False
        for (idx, op) in list(enumerate(block.ops)):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                is_clip_grad_by_global_norm = True
                break
        if not is_clip_grad_by_global_norm:
            return
        removed_op_idx = set()
        removed_tmp_var = set()
        for (idx, op) in list(enumerate(block.ops)):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                break
            for input_name in op.input_arg_names:
                input_var = block.var(input_name)
                if mp_rank >= 1 and (not (hasattr(input_var, 'is_distributed') and input_var.is_distributed)):
                    removed_op_idx.add(idx)
                    for output_name in op.output_arg_names:
                        removed_tmp_var.add(output_name)
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if not self._is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                block._remove_op(idx, sync=False)
        for var_name in removed_tmp_var:
            block._remove_var(var_name, sync=False)
        for (idx, op) in list(enumerate(block.ops)):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                sum_rst_var = block.var(op.output_arg_names[0])
                if mp_rank >= 1:
                    reserved_vars = []
                    for input_name in op.input_arg_names:
                        if input_name not in removed_tmp_var:
                            reserved_vars.append(input_name)
                    if len(reserved_vars) > 0:
                        op.desc.set_input('X', reserved_vars)
                    else:
                        namescope = op.attr('op_namescope')
                        block._remove_op(idx, sync=False)
                        fill_constant_op = block._insert_op_without_sync(idx, type='fill_constant', inputs={}, outputs={'Out': sum_rst_var}, attrs={'shape': sum_rst_var.shape, 'dtype': sum_rst_var.dtype, 'value': 0.0, OP_ROLE_KEY: OpRole.Optimize})
                        fill_constant_op._set_attr('op_namescope', namescope)
                self._insert_allreduce(block, ring_ids, idx, sum_rst_var)
                break

    @staticmethod
    def _insert_allreduce(block, ring_ids, idx, var):
        if False:
            i = 10
            return i + 15
        for ring_id in ring_ids:
            if ring_id == -1:
                continue
            idx = idx + 1
            block._insert_op_without_sync(idx, type='c_allreduce_sum', inputs={'X': var}, outputs={'Out': var}, attrs={'ring_id': ring_id, 'op_namescope': '/gradient_clip_model_parallelism', 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Optimize})