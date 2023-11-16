from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution

class StochasticPSpace(ProductPSpace):
    """
    Represents probability space of stochastic processes
    and their random variables. Contains mechanics to do
    computations for queries of stochastic processes.

    Explanation
    ===========

    Initialized by symbol, the specific process and
    distribution(optional) if the random indexed symbols
    of the process follows any specific distribution, like,
    in Bernoulli Process, each random indexed symbol follows
    Bernoulli distribution. For processes with memory, this
    parameter should not be passed.
    """

    def __new__(cls, sym, process, distribution=None):
        if False:
            i = 10
            return i + 15
        sym = _symbol_converter(sym)
        from sympy.stats.stochastic_process_types import StochasticProcess
        if not isinstance(process, StochasticProcess):
            raise TypeError('`process` must be an instance of StochasticProcess.')
        if distribution is None:
            distribution = Distribution()
        return Basic.__new__(cls, sym, process, distribution)

    @property
    def process(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The associated stochastic process.\n        '
        return self.args[1]

    @property
    def domain(self):
        if False:
            i = 10
            return i + 15
        return ProductDomain(self.process.index_set, self.process.state_space)

    @property
    def symbol(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[0]

    @property
    def distribution(self):
        if False:
            while True:
                i = 10
        return self.args[2]

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Transfers the task of handling queries to the specific stochastic\n        process because every process has their own logic of handling such\n        queries.\n        '
        return self.process.probability(condition, given_condition, evaluate, **kwargs)

    def compute_expectation(self, expr, condition=None, evaluate=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transfers the task of handling queries to the specific stochastic\n        process because every process has their own logic of handling such\n        queries.\n        '
        return self.process.expectation(expr, condition, evaluate, **kwargs)