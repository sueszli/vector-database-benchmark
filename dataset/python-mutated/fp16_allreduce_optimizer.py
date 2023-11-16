import paddle
from paddle.framework import core
from paddle.utils import unique_name
from .meta_optimizer_base import MetaOptimizerBase
__all__ = []

class FP16AllReduceOptimizer(MetaOptimizerBase):

    def __init__(self, optimizer):
        if False:
            while True:
                i = 10
        super().__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = ['LarsOptimizer', 'LambOptimizer', 'RecomputeOptimizer', 'LocalSGDOptimizer', 'GradientMergeOptimizer', 'AdaptiveLocalSGDOptimizer']
        self.meta_optimizers_black_list = ['DGCOptimizer']

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer, user_defined_strategy):
        if False:
            return 10
        super()._set_basic_info(loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _can_apply(self):
        if False:
            print('Hello World!')
        if not self.role_maker._is_collective:
            return False
        if self.user_defined_strategy.fp16_allreduce:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        if False:
            i = 10
            return i + 15
        dist_strategy.fp16_allreduce = False

    def _enable_strategy(self, dist_strategy, context=None):
        if False:
            return 10
        dist_strategy.fp16_allreduce = True

    @staticmethod
    def fp16_compression(param_and_grads):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compress fp32 gradients to fp16 during allreduce.\n        '
        op_maker = core.op_proto_and_checker_maker
        new_param_and_grads = []
        for (param, grad) in param_and_grads:
            if grad is None or grad.dtype != core.VarDesc.VarType.FP32:
                new_param_and_grads.append((param, grad, False))
                continue
            op = grad.op
            block = grad.block
            var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
            if param.name not in var_attr:
                new_param_and_grads.append((param, grad, False))
                continue
            var_attr.remove(param.name)
            var_attr.remove(grad.name)
            if len(var_attr) > 1:
                op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
            else:
                op._remove_attr(op_maker.kOpRoleVarAttrName())
            new_grad = block.create_var(name=unique_name.generate(grad.name + '.cast_fp16'), dtype=core.VarDesc.VarType.FP16, persistable=False, stop_gradient=True)
            with block.program._backward_role_guard():
                cast_op = block.append_op(type='cast', inputs={'X': grad}, outputs={'Out': new_grad}, attrs={'in_dtype': core.VarDesc.VarType.FP32, 'out_dtype': core.VarDesc.VarType.FP16}, stop_gradient=True)
                backward = op_maker.OpRole.Backward
                cast_op._set_attr(op_maker.kOpRoleAttrName(), backward)
                cast_op._set_attr(op_maker.kOpRoleVarAttrName(), [param.name, new_grad.name])
                new_grad.op = cast_op
            new_param_and_grads.append((param, new_grad, True))
        ret_param_and_grads = []
        for (param, grad, cast) in new_param_and_grads:
            if not cast:
                ret_param_and_grads.append((param, grad))
                continue
            block = grad.block
            new_grad = block.create_var(name=unique_name.generate(grad.name + '.cast_fp32'), dtype=core.VarDesc.VarType.FP32, persistable=False, stop_gradient=True)
            with block.program._optimized_guard([param, grad]), paddle.static.name_scope('fp16_allreduce'):
                cast_op = block.append_op(type='cast', inputs={'X': grad}, outputs={'Out': new_grad}, attrs={'in_dtype': core.VarDesc.VarType.FP16, 'out_dtype': core.VarDesc.VarType.FP32}, stop_gradient=True)
            ret_param_and_grads.append((param, new_grad))
        return ret_param_and_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        if False:
            i = 10
            return i + 15
        new_params_grads = self.fp16_compression(params_grads)
        return self.inner_opt._apply_optimize(loss, startup_program=startup_program, params_grads=new_params_grads)