from ..utils import is_dim_shard
from .common import DistributedOperatorImpl, DistributedOperatorImplContainer, register_distributed_operator_impl, register_distributed_operator_impl_container
from .dist_default import DistributedDefaultImpl0

class DistributedShape(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(op_type)
register_distributed_operator_impl_container(DistributedShape('shape'))

class DistributedShapeImpl(DistributedOperatorImpl):

    def __init__(self, name):
        if False:
            return 10
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        if False:
            return 10
        return True

    def is_output_compatible(self, dist_op):
        if False:
            return 10
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        assert len(out_dims_mapping) == 1
        if is_dim_shard(out_dims_mapping[0]):
            return False
        return True

    def is_auto_compatible(self, dist_op):
        if False:
            i = 10
            return i + 15
        if not self.is_input_compatible(dist_op) or not self.is_output_compatible(dist_op):
            return False
        return True

    def update_dims_mapping(self, dist_op):
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def forward(ctx, *args, **kwargs):
        if False:
            while True:
                i = 10
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        if False:
            return 10
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)
register_distributed_operator_impl('shape', DistributedShapeImpl('shape'))