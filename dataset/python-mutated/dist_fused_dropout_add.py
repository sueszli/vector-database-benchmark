import logging
import paddle
from paddle.base.log_helper import get_logger
from paddle.framework import core
from paddle.utils import unique_name
from ...random import determinate_rng, is_enable_auto_rand_ctrl
from ..utils import naive_set_dist_op_attr_for_program_by_mesh_and_mapping, set_var_dist_attr
from .common import DistributedOperatorImplContainer, register_distributed_operator_impl, register_distributed_operator_impl_container
from .dist_eltwise import DistributedDefaultImpl0, DistributedElementwiseImpl0
_logger = get_logger(__name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

class DistributedDropout(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(op_type)
register_distributed_operator_impl_container(DistributedDropout('fused_dropout_add'))

class DistributedDropoutImpl0(DistributedElementwiseImpl0):

    def __init__(self, name):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        return True

    @staticmethod
    def forward(ctx, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        if is_enable_auto_rand_ctrl() and (not op_dist_attr.is_recompute):
            assert op_dist_attr is not None, f"forward op [{str(src_op)}] don't have dist attribute !"
            assert 'seed_tensor' in kwargs, 'input [{}] is not given'.format('seed_tensor')
            if src_op.has_attr('fix_seed') and src_op.attr('fix_seed') and src_op.has_attr('seed') and src_op.attr('seed'):
                _logger.info(f'Auto Parallel Random Control Skipped Since manul seed is set by user: {src_op}')
            elif rank_id not in op_dist_attr.process_mesh.process_ids:
                pass
            elif len(kwargs['seed_tensor']) > 0 or len(src_op.input('seed_tensor')) > 0:
                seed_var_name = kwargs['seed_tensor'][0]
                if seed_var_name.startswith('rc_seed'):
                    pre_op = main_block.ops[-1]
                    assert pre_op.type == 'seed' and len(pre_op.attr('rng_name')) == 0, f'found exception op {str(pre_op)}'
                    X_var = main_block._var_recursive(kwargs['x'][0])
                    X_dims_mapping = op_dist_attr.get_input_dims_mapping(X_var.name)
                    process_mesh = op_dist_attr.process_mesh
                    rng_name = determinate_rng(rank_id, X_dims_mapping, process_mesh)
                    pre_op._set_attr('rng_name', rng_name)
                    pre_op._set_attr('deterministic', True)
                    pre_op._set_attr('force_cpu', True)
                else:
                    _logger.info(f'Auto Parallel Random Control Skipped Since manul seed is set by user: {src_op}')
            else:
                X_var = main_block._var_recursive(kwargs['x'][0])
                X_dims_mapping = op_dist_attr.get_input_dims_mapping(X_var.name)
                process_mesh = op_dist_attr.process_mesh
                rng_name = determinate_rng(rank_id, X_dims_mapping, process_mesh)
                assert rng_name is not None and rng_name != ''
                seed_var = main_block.create_var(name=unique_name.generate_with_ignorable_key('.'.join(['tensor_parallel_seed', 'tmp'])), dtype=paddle.int32, type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
                seed_var_dims_mapping = [-1]
                seed_var_dist_attr = set_var_dist_attr(ctx, seed_var, seed_var_dims_mapping, process_mesh)
                seed_op = main_block.append_op(type='seed', outputs={'Out': seed_var}, attrs={'deterministic': True, 'rng_name': rng_name, 'force_cpu': True})
                seed_op._set_attr('op_namescope', 'auto_tensor_parallel_seed')
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(seed_op, process_mesh, seed_var_dims_mapping, ctx)
                src_op.desc.set_input('seed_tensor', [seed_var.name])
                src_op._remove_attr('fix_seed')
                src_op._remove_attr('seed')
                op_dist_attr.set_input_dist_attr(seed_var.name, seed_var_dist_attr)
                kwargs['seed_tensor'] = [seed_var.name]
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        if False:
            while True:
                i = 10
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)
register_distributed_operator_impl('fused_dropout_add', DistributedDropoutImpl0('random_control'))