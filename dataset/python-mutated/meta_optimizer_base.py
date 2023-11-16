from paddle.optimizer import Optimizer
__all__ = []

class MetaOptimizerBase(Optimizer):

    def __init__(self, optimizer):
        if False:
            i = 10
            return i + 15
        self.inner_opt = optimizer
        self._learning_rate = self.inner_opt._learning_rate
        self._learning_rate_map = self.inner_opt._learning_rate_map
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _set_auxiliary_var(self, key, val):
        if False:
            i = 10
            return i + 15
        super()._set_auxiliary_var(key, val)
        self.inner_opt._set_auxiliary_var(key, val)

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer, user_defined_strategy):
        if False:
            i = 10
            return i + 15
        self.loss = loss
        self.role_maker = role_maker
        self.user_defined_optimizer = user_defined_optimizer
        self.user_defined_strategy = user_defined_strategy

    def _update_inner_optimizer(self, optimizer):
        if False:
            i = 10
            return i + 15
        self.inner_opt = optimizer

    def _can_apply(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def _is_graph_out(self):
        if False:
            print('Hello World!')
        return False

    def _can_update(self, optimizer):
        if False:
            i = 10
            return i + 15
        if str(optimizer.__class__.__name__) in self.meta_optimizers_white_list:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'you should implement disable strategy in {type(self).__name__}')

    def _enable_strategy(self, dist_strategy, context=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'you should implement enable strategy in {type(self).__name__}')

    def apply_gradients(self, params_grads):
        if False:
            print('Hello World!')
        return self.inner_opt.apply_gradients(params_grads=params_grads)

    def backward(self, loss, startup_program=None, parameter_list=None, no_grad_set=None, callbacks=None):
        if False:
            i = 10
            return i + 15
        return self.inner_opt.backward(loss, startup_program, parameter_list, no_grad_set, callbacks)

    def apply_optimize(self, loss, startup_program, params_grads):
        if False:
            while True:
                i = 10
        return self.inner_opt._apply_optimize(loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            print('Hello World!')
        params_grads = self.backward(loss, startup_program=startup_program, parameter_list=parameter_list, no_grad_set=no_grad_set)
        optimize_ops = self.apply_optimize(loss, startup_program=startup_program, params_grads=params_grads)
        return (optimize_ops, params_grads)

    def minimize(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            print('Hello World!')
        (optimize_ops, params_grads) = self.minimize_impl(loss, startup_program, parameter_list, no_grad_set)
        return (optimize_ops, params_grads)