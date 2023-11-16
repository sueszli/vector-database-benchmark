from ..utils import compute_compatible_and_update_dim_mapping
from .common import DistributedOperatorImpl, DistributedOperatorImplContainer
from .dist_default import DistributedDefaultImpl0

class DistributedAssign(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            print('Hello World!')
        super().__init__(op_type)

class DistributedAssignImpl(DistributedOperatorImpl):

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
            while True:
                i = 10
        return True

    def is_auto_compatible(self, dist_op):
        if False:
            print('Hello World!')
        if not self.is_input_compatible(dist_op) or not self.is_output_compatible(dist_op):
            return False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if x_dims_mapping != out_dims_mapping:
            return False
        return True

    def update_dims_mapping(self, dist_op):
        if False:
            for i in range(10):
                print('nop')
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        for i in range(len(x_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping([x_dims_mapping, out_dims_mapping], [i, i])
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
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        if False:
            return 10
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)