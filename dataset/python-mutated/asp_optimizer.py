from paddle.incubate.asp import ASPHelper
from .meta_optimizer_base import MetaOptimizerBase
__all__ = []

class ASPOptimizer(MetaOptimizerBase):

    def __init__(self, optimizer):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = ['AMPOptimizer', 'LarsOptimizer', 'LambOptimizer', 'RecomputeOptimizer', 'GradientMergeOptimizer']
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer, user_defined_strategy):
        if False:
            print('Hello World!')
        super()._set_basic_info(loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _can_apply(self):
        if False:
            i = 10
            return i + 15
        if not self.role_maker._is_collective:
            return False
        if self.user_defined_strategy.asp:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        if False:
            while True:
                i = 10
        dist_strategy.asp = False

    def _enable_strategy(self, dist_strategy, context):
        if False:
            return 10
        dist_strategy.asp = True

    def minimize_impl(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            i = 10
            return i + 15
        (optimize_ops, params_grads) = ASPHelper._minimize(self.inner_opt, loss, startup_program=startup_program, parameter_list=parameter_list, no_grad_set=no_grad_set)
        return (optimize_ops, params_grads)