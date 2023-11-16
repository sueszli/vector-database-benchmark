from ..meta_optimizers import *
__all__ = []
meta_optimizer_names = list(filter(lambda name: name.endswith('Optimizer'), dir()))
meta_optimizer_names.remove('HybridParallelOptimizer')
meta_optimizer_names.remove('HeterParallelOptimizer')
meta_optimizer_names.remove('DGCMomentumOptimizer')

class MetaOptimizerFactory:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def _get_valid_meta_optimizers(self, user_defined_optimizer):
        if False:
            for i in range(10):
                print('nop')
        opt_list = []
        for opt_name in meta_optimizer_names:
            opt_list.append(globals()[opt_name](user_defined_optimizer))
        return opt_list