import copy
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
from ..completion import get_phi_spmd_rule
from ..dist_attribute import OperatorDistAttr
from ..process_group import new_process_group
from ..utils import get_dist_tensor_spec, is_dim_shard, set_dist_op_desc_original_id
from .common import DistributedOperatorImpl, DistributedOperatorImplContainer, get_default_distributed_operator_impl, register_distributed_operator_impl, register_distributed_operator_impl_container, update_op_dims_mapping

class DistributedReduceSum(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            return 10
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        if False:
            return 10
        op_desc = dist_op.serial_op.desc
        assert len(op_desc.input_arg_names()) == 1, 'reduce_sum op [{}] has [{}] inputs'.format(op_desc.type, len(op_desc.input_arg_names()))
        input_arg_name = op_desc.input_arg_names()[0]
        assert len(op_desc.output_arg_names()) == 1, 'reduce_sum op [{}] has [{}] outputs'.format(op_desc.type, len(op_desc.output_arg_names()))
        output_arg_name = op_desc.output_arg_names()[0]
        keep_dim = op_desc.attr('keep_dim')
        dims = op_desc.attr('dim')
        input_spec = get_dist_tensor_spec(dist_op, input_arg_name)
        output_spec = get_dist_tensor_spec(dist_op, output_arg_name, False)
        if len(dims) == 0:
            dims = list(range(len(input_spec.shape)))
        rule = get_phi_spmd_rule('reduce_sum')
        fw_results = rule.infer_forward(input_spec, dims, keep_dim)
        bw_results = rule.infer_backward(input_spec, output_spec, dims, keep_dim)
        changed = update_op_dims_mapping(dist_op, [input_arg_name], [output_arg_name], fw_results, bw_results)
        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        if False:
            return 10
        op_dist_attr = dist_op.dist_attr
        op_desc = dist_op.serial_op.desc
        input_name = op_desc.input_arg_names()[0]
        input_dims_mapping = copy.deepcopy(op_dist_attr.get_input_dims_mapping(input_name))
        axes = op_desc.attr('dim')
        op_dist_attr = dist_op.dist_attr
        reverted = False

        def is_partial_reduce(axes, dims_mapping):
            if False:
                while True:
                    i = 10
            if len(axes) != 0 and len(axes) < len(dims_mapping):
                for axis in axes:
                    if is_dim_shard(dims_mapping[axis]):
                        return True
            return False
        if is_partial_reduce(axes, input_dims_mapping):
            dist_op.dist_attr = original_op_dist_attr
            reverted = True
        else:
            default_impl = get_default_distributed_operator_impl()
            op_dist_attr.impl_type = default_impl.type
            op_dist_attr.impl_idx = default_impl.idx
        return reverted
register_distributed_operator_impl_container(DistributedReduceSum('reduce_sum'))

class DistributedReduceSumPrimtive(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            print('Hello World!')
        super().__init__(op_type)
register_distributed_operator_impl_container(DistributedReduceSumPrimtive('reduce_sum_p'))

class DistributedReduceSumPrimtiveImpl0(DistributedOperatorImpl):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        if False:
            print('Hello World!')
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        return len(op_desc.input_arg_names()) == 1

    def is_output_compatible(self, dist_op):
        if False:
            i = 10
            return i + 15
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        outputs = op_desc.output_arg_names()
        if len(outputs) != 1:
            return False
        output_name = outputs[0]
        output_var = dist_op.serial_op.block._var_recursive(output_name)
        if output_var.shape != ():
            return False
        return True

    def is_auto_compatible(self, dist_op):
        if False:
            return 10
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        return self.is_input_compatible(dist_op) and self.is_output_compatible(dist_op)

    def update_dims_mapping(self, dist_op):
        if False:
            return 10
        changed = False
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        if False:
            print('Hello World!')
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, f'input [{input_name}] is not given'
            assert len(kwargs[input_name]) == len(src_op.desc.input(input_name)), f'number of tensor for input [{input_name}] is not match'
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, f'input [{output_name}] is not given'
            assert len(kwargs[output_name]) == len(src_op.desc.output(output_name)), f'number of tensor for input [{output_name}] is not match'
        dist_op = main_block.append_op(type='nop')
        dist_op_desc = dist_op.desc
        dist_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])
        var_name = src_op.output_arg_names[0]
        sync_group = new_process_group(ctx.data_parallel_group)
        allreduce_op = main_block.append_op(type='c_allreduce_sum', inputs={'X': [var_name]}, outputs={'Out': [var_name]}, attrs={'ring_id': sync_group.id, 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Forward})
        var = main_block._var_recursive(var_name)
        tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(var)
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        new_op_attr = OperatorDistAttr()
        new_op_attr.process_mesh = op_dist_attr.process_mesh
        new_op_attr.set_output_dims_mapping(var.name, tensor_dist_attr.dims_mapping)
        new_op_attr.set_input_dims_mapping(var.name, tensor_dist_attr.dims_mapping)
        ctx.set_op_dist_attr_for_program(allreduce_op, new_op_attr)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise RuntimeError('primitive operator does NOT have backward function, op type: {}'.format(str(op.type)))
register_distributed_operator_impl('reduce_sum_p', DistributedReduceSumPrimtiveImpl0('batch_dimension_reduce_sum_p'))