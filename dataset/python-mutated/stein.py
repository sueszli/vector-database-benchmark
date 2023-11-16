import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pymc.pytensorf import floatX
from pymc.util import WithMemoization, locally_cachedmethod
from pymc.variational.opvi import node_property
from pymc.variational.test_functions import rbf
__all__ = ['Stein']

class Stein(WithMemoization):

    def __init__(self, approx, kernel=rbf, use_histogram=True, temperature=1):
        if False:
            while True:
                i = 10
        self.approx = approx
        self.temperature = floatX(temperature)
        self._kernel_f = kernel
        self.use_histogram = use_histogram

    @property
    def input_joint_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        if self.use_histogram:
            return self.approx.joint_histogram
        else:
            return self.approx.symbolic_random

    @node_property
    def approx_symbolic_matrices(self):
        if False:
            print('Hello World!')
        if self.use_histogram:
            return self.approx.collect('histogram')
        else:
            return self.approx.symbolic_randoms

    @node_property
    def dlogp(self):
        if False:
            for i in range(10):
                print('nop')
        logp = self.logp_norm.sum()
        grad = pt.grad(logp, self.approx_symbolic_matrices)

        def flatten2(tensor):
            if False:
                return 10
            return tensor.flatten(2)
        return pt.concatenate(list(map(flatten2, grad)), -1)

    @node_property
    def grad(self):
        if False:
            while True:
                i = 10
        n = floatX(self.input_joint_matrix.shape[0])
        temperature = self.temperature
        svgd_grad = self.density_part_grad / temperature + self.repulsive_part_grad
        return svgd_grad / n

    @node_property
    def density_part_grad(self):
        if False:
            return 10
        Kxy = self.Kxy
        dlogpdx = self.dlogp
        return pt.dot(Kxy, dlogpdx)

    @node_property
    def repulsive_part_grad(self):
        if False:
            for i in range(10):
                print('nop')
        t = self.approx.symbolic_normalizing_constant
        dxkxy = self.dxkxy
        return dxkxy / t

    @property
    def Kxy(self):
        if False:
            print('Hello World!')
        return self._kernel()[0]

    @property
    def dxkxy(self):
        if False:
            for i in range(10):
                print('nop')
        return self._kernel()[1]

    @node_property
    def logp_norm(self):
        if False:
            i = 10
            return i + 15
        sized_symbolic_logp = self.approx.sized_symbolic_logp
        if self.use_histogram:
            sized_symbolic_logp = graph_replace(sized_symbolic_logp, dict(zip(self.approx.symbolic_randoms, self.approx.collect('histogram'))), strict=False)
        return sized_symbolic_logp / self.approx.symbolic_normalizing_constant

    @locally_cachedmethod
    def _kernel(self):
        if False:
            return 10
        return self._kernel_f(self.input_joint_matrix)