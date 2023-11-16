from paddle.incubate.optimizer import GradientMergeOptimizer as GM
from .meta_optimizer_base import MetaOptimizerBase
__all__ = []

class GradientMergeOptimizer(MetaOptimizerBase):

    def __init__(self, optimizer):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(optimizer)
        self.inner_opt = optimizer
        self.wrapped_opt = None
        self.meta_optimizers_white_list = ['AMPOptimizer', 'LarsOptimizer', 'LambOptimizer', 'RecomputeOptimizer']
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer, user_defined_strategy):
        if False:
            while True:
                i = 10
        super()._set_basic_info(loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_wrapped_opt(self):
        if False:
            print('Hello World!')
        config = self.user_defined_strategy.gradient_merge_configs
        self.wrapped_opt = GM(self.inner_opt)
        self.wrapped_opt._set_k_steps(self.user_defined_strategy.gradient_merge_configs['k_steps'])
        self.wrapped_opt._set_avg(self.user_defined_strategy.gradient_merge_configs['avg'])

    def _can_apply(self):
        if False:
            while True:
                i = 10
        if not self.role_maker._is_collective:
            return False
        can_apply = self.user_defined_strategy.gradient_merge and self.user_defined_strategy.gradient_merge_configs['k_steps'] > 1
        return can_apply

    def _disable_strategy(self, dist_strategy):
        if False:
            for i in range(10):
                print('nop')
        dist_strategy.gradient_merge = False
        dist_strategy.gradient_merge_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        if False:
            return 10
        return

    def minimize_impl(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            while True:
                i = 10
        self._init_wrapped_opt()
        (optimize_ops, params_grads) = self.wrapped_opt.minimize(loss, startup_program, parameter_list, no_grad_set)
        return (optimize_ops, params_grads)