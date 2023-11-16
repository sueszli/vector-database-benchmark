from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole, is_optimizer_op
from paddle.framework import core
__all__ = []

class FP16Utils:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def is_fp16_cast_op(block, op, params):
        if False:
            i = 10
            return i + 15
        if op.type != 'cast':
            return False
        if is_optimizer_op(op):
            return False
        assert len(op.desc.input_arg_names()) == 1
        assert len(op.desc.output_arg_names()) == 1
        (input_name, output_name) = (op.desc.input_arg_names()[0], op.desc.output_arg_names()[0])
        if input_name not in params:
            return False
        input_var = block.var(input_name)
        output_var = block.var(output_name)
        if input_var.dtype != core.VarDesc.VarType.FP32 or output_var.dtype != core.VarDesc.VarType.FP16:
            return False
        return True

    @staticmethod
    def is_fp32_cast_op(block, op):
        if False:
            print('Hello World!')
        if op.type != 'cast':
            return False
        if not is_optimizer_op(op):
            return False
        assert len(op.desc.input_arg_names()) == 1
        assert len(op.desc.output_arg_names()) == 1
        (input_name, output_name) = (op.desc.input_arg_names()[0], op.desc.output_arg_names()[0])
        input_var = block.var(input_name)
        output_var = block.var(output_name)
        if input_var.dtype != core.VarDesc.VarType.FP16 or output_var.dtype != core.VarDesc.VarType.FP32:
            return False
        return True

    @staticmethod
    def remove_cast_op(block, params, segment, offset):
        if False:
            while True:
                i = 10
        inserted_op_num = 0
        for op_idx in reversed(range(offset + segment._start_idx, offset + segment._end_idx)):
            op = block.ops[op_idx]
            if FP16Utils.is_fp16_cast_op(block, op, params):
                block._remove_op(op_idx, sync=False)
                inserted_op_num -= 1
        block._sync_with_cpp()
        return inserted_op_num

    @staticmethod
    def prune_fp16(block, shard, reduced_grads_to_param, ring_ids):
        if False:
            i = 10
            return i + 15
        '\n        1. prune all cast_fp16_to_fp32 ops if the param not belongs to this shard\n        2. revise amp inifine grad checking for sharding\n        '
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if not FP16Utils.is_fp32_cast_op(block, op):
                continue
            output_name = op.desc.output_arg_names()[0]
            param_name = output_name.strip('@GRAD@MERGED') if '@MERGED' in output_name else output_name.strip('@GRAD')
            if param_name not in shard.global_params:
                raise ValueError(f"Output 'X' of cast_op must be a grad ofmodel param, but {output_name} is not a grad")
            if output_name in reduced_grads_to_param:
                continue
            if shard.has_param(param_name):
                continue
            block._remove_op(idx, sync=False)
            block._remove_var(output_name, sync=False)
        block._sync_with_cpp()
        update_loss_scaling_op_idx = -1
        inf_var_name = ''
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if op.type == 'update_loss_scaling':
                update_loss_scaling_op_idx = idx
                inf_var_name = op.desc.input('FoundInfinite')[0]
            if op.type in ['check_finite_and_unscale', 'update_loss_scaling']:
                reversed_x = []
                reversed_x_paramname = []
                for input_name in op.desc.input('X'):
                    if '@MERGED' in input_name:
                        param_name = input_name.strip('@GRAD@MERGED')
                    else:
                        param_name = input_name.strip('@GRAD')
                    if param_name not in shard.global_params:
                        raise ValueError(f"Input 'X' of check_finite_and_unscale mustbe grads, but {input_name} is not a grad")
                    if shard.has_param(param_name):
                        reversed_x.append(input_name)
                        reversed_x_paramname.append(param_name)
                op.desc.set_input('X', reversed_x)
                op.desc.set_output('Out', reversed_x)
                to_check_param = set(reversed_x_paramname)
                should_check_param = set(shard.global_params).intersection({param for (param, worker_idx) in shard.global_param2device.items() if worker_idx == shard.worker_idx})
                assert to_check_param == should_check_param, 'amp                     check_finite_and_unscale checking miss [{}] and got unexpected [{}]'.format(should_check_param - to_check_param, to_check_param - should_check_param)
        if update_loss_scaling_op_idx == -1:
            return
        inf_var = block.var(inf_var_name)
        inf_var_int32 = block.create_var(name=inf_var_name + '@cast_int32', shape=inf_var.shape, dtype=core.VarDesc.VarType.INT32)
        block._insert_op_without_sync(update_loss_scaling_op_idx, type='cast', inputs={'X': inf_var}, outputs={'Out': inf_var_int32}, attrs={'in_dtype': inf_var.dtype, 'out_dtype': inf_var_int32.dtype, OP_ROLE_KEY: OpRole.Optimize})
        update_loss_scaling_op_idx += 1
        for ring_id in ring_ids:
            if ring_id == -1:
                continue
            block._insert_op_without_sync(update_loss_scaling_op_idx, type='c_allreduce_max', inputs={'X': inf_var_int32}, outputs={'Out': inf_var_int32}, attrs={'ring_id': ring_id, 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Optimize})
            update_loss_scaling_op_idx += 1
        block._insert_op_without_sync(update_loss_scaling_op_idx, type='cast', inputs={'X': inf_var_int32}, outputs={'Out': inf_var}, attrs={'in_dtype': inf_var_int32.dtype, 'out_dtype': inf_var.dtype, OP_ROLE_KEY: OpRole.Optimize})
        update_loss_scaling_op_idx += 1
        block._sync_with_cpp()

    @staticmethod
    def sync_amp_check_nan_inf(block, ring_ids):
        if False:
            for i in range(10):
                print('nop')
        update_loss_scaling_op_idx = -1
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if op.type == 'update_loss_scaling':
                update_loss_scaling_op_idx = idx
                inf_var_name = op.desc.input('FoundInfinite')[0]
                break
        if update_loss_scaling_op_idx == -1:
            return
        inf_var = block.var(inf_var_name)
        inf_var_int32 = block.create_var(name=inf_var_name + '@cast_int32', shape=inf_var.shape, dtype=core.VarDesc.VarType.INT32)
        block._insert_op_without_sync(update_loss_scaling_op_idx, type='cast', inputs={'X': inf_var}, outputs={'Out': inf_var_int32}, attrs={'in_dtype': inf_var.dtype, 'out_dtype': inf_var_int32.dtype, OP_ROLE_KEY: OpRole.Optimize})
        update_loss_scaling_op_idx += 1
        for ring_id in ring_ids:
            if ring_id == -1:
                continue
            block._insert_op_without_sync(update_loss_scaling_op_idx, type='c_allreduce_max', inputs={'X': inf_var_int32}, outputs={'Out': inf_var_int32}, attrs={'ring_id': ring_id, 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Optimize})
            update_loss_scaling_op_idx += 1
        block._insert_op_without_sync(update_loss_scaling_op_idx, type='cast', inputs={'X': inf_var_int32}, outputs={'Out': inf_var}, attrs={'in_dtype': inf_var_int32.dtype, 'out_dtype': inf_var.dtype, OP_ROLE_KEY: OpRole.Optimize})
        update_loss_scaling_op_idx += 1
        block._sync_with_cpp()