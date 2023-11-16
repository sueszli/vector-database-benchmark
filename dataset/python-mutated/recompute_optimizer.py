from paddle.incubate.optimizer import RecomputeOptimizer as RO
from .meta_optimizer_base import MetaOptimizerBase
__all__ = []

class RecomputeOptimizer(MetaOptimizerBase):

    def __init__(self, optimizer):
        if False:
            i = 10
            return i + 15
        super().__init__(optimizer)
        self.inner_opt = optimizer
        self.wrapped_opt = None
        self.meta_optimizers_white_list = ['LarsOptimizer', 'LambOptimizer', 'DGCOptimizer']
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer, user_defined_strategy):
        if False:
            while True:
                i = 10
        super()._set_basic_info(loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_wrapped_opt(self):
        if False:
            i = 10
            return i + 15
        if self.wrapped_opt is not None:
            return
        configs = self.user_defined_strategy.recompute_configs
        self.wrapped_opt = RO(self.inner_opt)
        self.wrapped_opt._set_checkpoints(list(configs['checkpoints']))
        if configs['enable_offload']:
            self.wrapped_opt._enable_offload()
            checkpoint_shapes = list(configs['checkpoint_shape'])
            self.wrapped_opt.checkpoint_shape = checkpoint_shapes

    def _can_apply(self):
        if False:
            i = 10
            return i + 15
        if not self.role_maker._is_collective:
            return False
        if self.user_defined_strategy.recompute:
            if len(self.user_defined_strategy.recompute_configs['checkpoints']) == 0:
                return False
            else:
                return True

    def _disable_strategy(self, dist_strategy):
        if False:
            while True:
                i = 10
        dist_strategy.recompute = False
        dist_strategy.recompute_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        if False:
            print('Hello World!')
        return

    def backward(self, loss, startup_program=None, parameter_list=None, no_grad_set=None, callbacks=None):
        if False:
            while True:
                i = 10
        self._init_wrapped_opt()
        return self.wrapped_opt.backward(loss, startup_program, parameter_list, no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        if False:
            while True:
                i = 10
        return self.wrapped_opt.apply_gradients(params_grads=params_grads)

    def apply_optimize(self, loss, startup_program, params_grads):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapped_opt.apply_optimize(loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            print('Hello World!')
        self._init_wrapped_opt()
        (optimize_ops, params_grads) = self.wrapped_opt.minimize(loss, startup_program, parameter_list, no_grad_set)
        return (optimize_ops, params_grads)