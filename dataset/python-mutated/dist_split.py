from ..completion import get_phi_spmd_rule
from ..utils import compute_compatible_and_update_dim_mapping, get_dist_tensor_spec, is_dim_shard
from .common import DistributedOperatorImpl, DistributedOperatorImplContainer, get_default_distributed_operator_impl, register_distributed_operator_impl, register_distributed_operator_impl_container, update_op_dims_mapping
from .dist_default import DistributedDefaultImpl0

class DistributedSplit(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        if False:
            print('Hello World!')
        op_desc = dist_op.serial_op.desc
        x_name = op_desc.input('X')[0]
        assert len(op_desc.input('AxisTensor')) == 0, 'Attribute AxisTensor is not supported by dist split.'
        assert len(op_desc.input('SectionsTensorList')) == 0, 'Attribute SectionsTensorList is not supported by dist split.'
        output_arg_names = op_desc.output('Out')
        num = op_desc.attr('num')
        sections = op_desc.attr('sections')
        if num is not None:
            assert sections is None or len(sections) == 0, f'Both Attributes of num: {num} and sections: {sections} are specified.'
            first_attr = num
            rule_type = 'split_with_num'
        else:
            assert num is None, f'Both Attributes of num: {num} and sections: {sections} are specified.'
            first_attr = sections
            rule_type = 'split'
        axis = op_desc.attr('axis')
        x_spec = get_dist_tensor_spec(dist_op, x_name)
        num_outputs = len(output_arg_names)
        output_specs = []
        for i in range(num_outputs):
            output_specs.append(get_dist_tensor_spec(dist_op, output_arg_names[i], False))
        rule = get_phi_spmd_rule(rule_type)
        fw_results = rule.infer_forward(x_spec, first_attr, axis)
        bw_results = rule.infer_backward(x_spec, output_specs, first_attr, axis)
        changed = update_op_dims_mapping(dist_op, [x_name], output_arg_names, fw_results, bw_results)
        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        if False:
            print('Hello World!')
        op_dist_attr = dist_op.dist_attr
        default_impl = get_default_distributed_operator_impl()
        op_dist_attr.impl_type = default_impl.type
        op_dist_attr.impl_idx = default_impl.idx
        return False
register_distributed_operator_impl_container(DistributedSplit('split'))
register_distributed_operator_impl_container(DistributedSplit('split_with_num'))

class DistributedSplitImpl(DistributedOperatorImpl):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        if False:
            i = 10
            return i + 15
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        axis = op_desc.attr('axis')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        if is_dim_shard(x_dims_mapping[axis]):
            return False
        return True

    def is_output_compatible(self, dist_op):
        if False:
            while True:
                i = 10
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_names = op_desc.output('Out')
        axis = op_desc.attr('axis')
        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            if is_dim_shard(out_dims_mapping[axis]):
                return False
        return True

    def is_compatible(self, dist_op):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_input_compatible(dist_op) or not self.is_output_compatible(dist_op):
            return False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        axis = op_desc.attr('axis')
        out_names = op_desc.output('Out')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        for out_name in out_names:
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
        out_names = op_desc.output('Out')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            for i in range(len(x_dims_mapping)):
                dim_changed = compute_compatible_and_update_dim_mapping([x_dims_mapping, out_dims_mapping], [i, i])
                if dim_changed:
                    changed = True
                    op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)
        if changed:
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
        return changed

    def is_auto_compatible(self, dist_op):
        if False:
            while True:
                i = 10
        if not self.is_input_compatible(dist_op) or not self.is_output_compatible(dist_op) or (not self.is_compatible(dist_op)):
            return False
        return True

    @staticmethod
    def forward(ctx, *args, **kwargs):
        if False:
            return 10
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        if False:
            while True:
                i = 10
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)
register_distributed_operator_impl('split', DistributedSplitImpl('replicate_in_axis'))