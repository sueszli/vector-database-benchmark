from torch.fx import Tracer

class TestAllLeafModulesTracer(Tracer):

    def is_leaf_module(self, m, qualname):
        if False:
            i = 10
            return i + 15
        return True