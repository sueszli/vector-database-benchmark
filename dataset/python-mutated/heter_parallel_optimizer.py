import paddle.autograd as imperative_base
from paddle import framework
__all__ = []

def _obtain_optimizer_parameters_list(optimizer):
    if False:
        i = 10
        return i + 15
    if getattr(optimizer, '_param_groups', None) and isinstance(optimizer._param_groups[0], dict):
        parameters_list = []
        for group in optimizer._param_groups:
            for param in group['params']:
                parameters_list.append(param)
    else:
        parameters_list = list(optimizer._parameter_list)
    return parameters_list

class HeterParallelOptimizer:

    def __init__(self, optimizer, strategy):
        if False:
            return 10
        self._inner_opt = optimizer
        self._strategy = strategy

    @imperative_base.no_grad()
    @framework.dygraph_only
    def step(self):
        if False:
            return 10
        parameters_list = _obtain_optimizer_parameters_list(self._inner_opt)
        self._inner_opt.step()

    @imperative_base.no_grad()
    def minimize(self, loss, startup_program=None, parameters=None, no_grad_set=None):
        if False:
            for i in range(10):
                print('nop')
        parameter_list = parameters if parameters else self._inner_opt._parameter_list
        return self._inner_opt.minimize(loss, startup_program, parameter_list, no_grad_set)

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        return getattr(self._inner_opt, item)