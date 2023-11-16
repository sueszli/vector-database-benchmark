from paddle.distributed.fleet.meta_optimizers.common import OpRole
from ..cost import _g_op_cost_factory, build_comp_costs_from_descs, build_comp_desc_from_dist_op, build_dp_costs
from ..utils import compute_compatible_and_update_dim_mapping
from .common import DistributedOperatorImpl, DistributedOperatorImplContainer, is_parameter_related
from .dist_default import DistributedDefaultImpl0

class DistributedScale(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            i = 10
            return i + 15
        super().__init__(op_type)

class DistributedScaleImpl(DistributedOperatorImpl):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        if False:
            print('Hello World!')
        return True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        if False:
            i = 10
            return i + 15
        'Calculate the cost by the op role.'
        cost = None
        if int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
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
        cost_mapping = build_comp_costs_from_descs(_g_op_cost_factory[op_type], ctx, processes, desc_mapping, cluster)
        res_cost = [cost_mapping]
        return res_cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        if False:
            i = 10
            return i + 15
        res = []
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op, dist_context=ctx)
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        backward_op = dist_op.serial_op
        op_type = backward_op.type
        cost_mapping = build_comp_costs_from_descs(_g_op_cost_factory[op_type], ctx, processes, desc_mapping, cluster)
        res.append(cost_mapping)
        main_block = backward_op.block
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if '@GRAD' not in varname and (not is_parameter_related(varname, main_block)):
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)
                    mesh_shape = process_mesh.shape
                    batch_size_axis = var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                        need_gradient_allreduce = True
                        break
        if need_gradient_allreduce:
            for input_name in backward_op.desc.input_names():
                for varname in backward_op.desc.input(input_name):
                    if '@GRAD' not in varname and is_parameter_related(varname, main_block):
                        var_dim_mapping = dist_attr.get_input_dims_mapping(varname)
                        mesh_shape = process_mesh.shape
                        parallel_axis = batch_size_axis
                        attrs = {'use_calc_stream': True}
                        var_names = [varname + '@GRAD']
                        build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis, cluster)
        return res

    def is_output_compatible(self, dist_op):
        if False:
            return 10
        return True

    def is_auto_compatible(self, dist_op):
        if False:
            i = 10
            return i + 15
        if not self.is_input_compatible(dist_op) or not self.is_output_compatible(dist_op):
            return False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        in_dims_mappings = []
        for in_name in op_desc.input_arg_names():
            in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)
            in_dims_mappings.append(in_dims_mapping)
        for x_dims_mapping in in_dims_mappings:
            if x_dims_mapping != out_dims_mapping:
                return False
        return True

    def update_dims_mapping(self, dist_op):
        if False:
            i = 10
            return i + 15
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
                op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
                op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)
                changed = True
        return changed

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