from paddle.distributed.fleet.meta_optimizers.common import OpRole
from ..cost import FillConstantBatchSizeLikeOpCost, build_comp_costs_from_descs, build_comp_desc_from_dist_op
from ..utils import compute_compatible_and_update_dim_mapping
from .common import DistributedOperatorImpl, DistributedOperatorImplContainer, register_distributed_operator_impl, register_distributed_operator_impl_container
from .dist_default import DistributedDefaultImpl0

class DistributedFillConstantBatchSizeLike(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(op_type)
register_distributed_operator_impl_container(DistributedFillConstantBatchSizeLike('fill_constant_batch_size_like'))

class DistributedFillConstantBatchSizeLikeImpl0(DistributedOperatorImpl):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        if False:
            while True:
                i = 10
        cost = None
        if int(op_role) == int(OpRole.Backward):
            raise ValueError('The fill_constant_batch_size_like has no grad op.')
        else:
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        if False:
            while True:
                i = 10
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op, dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.process_ids
        op_type = dist_op.serial_op.type
        cost_mapping = build_comp_costs_from_descs(FillConstantBatchSizeLikeOpCost, ctx, processes, desc_mapping, cluster)
        res_cost = [cost_mapping]
        return res_cost

    def is_input_compatible(self, dist_op):
        if False:
            return 10
        return True

    def is_output_compatible(self, dist_op):
        if False:
            print('Hello World!')
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        shape_list = op_desc.attr('shape')
        if len(shape_list) != len(out_dims_mapping):
            return False
        return True

    def is_auto_compatible(self, dist_op):
        if False:
            return 10
        if not self.is_input_compatible(dist_op) or not self.is_output_compatible(dist_op):
            return False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        in_name = op_desc.input('Input')[0]
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)
        return out_dims_mapping[0] == in_dims_mapping[0]

    def update_dims_mapping(self, dist_op):
        if False:
            return 10
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('Input')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        dim_changed = compute_compatible_and_update_dim_mapping([x_dims_mapping, out_dims_mapping], [0, 0])
        if dim_changed:
            changed = True
        if changed:
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
            op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        kwargs: inputname_mapping & outputname_mapping\n        '
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)
register_distributed_operator_impl('fill_constant_batch_size_like', DistributedFillConstantBatchSizeLikeImpl0('fill_by_shape'))