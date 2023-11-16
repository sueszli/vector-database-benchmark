import abc
from collections import namedtuple
from typing import Optional
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
__all__ = ['PassResult', 'PassBase']

@compatibility(is_backward_compatible=False)
class PassResult(namedtuple('PassResult', ['graph_module', 'modified'])):
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """

    def __new__(cls, graph_module, modified):
        if False:
            while True:
                i = 10
        return super().__new__(cls, graph_module, modified)

@compatibility(is_backward_compatible=False)
class PassBase(abc.ABC):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __call__(self, graph_module: GraphModule) -> Optional[PassResult]:
        if False:
            return 10
        '\n        Runs the precondition check, the pass itself, and the postcondition check.\n        '
        self.requires(graph_module)
        res = self.call(graph_module)
        self.ensures(graph_module)
        return res

    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        if False:
            return 10
        '\n        The pass that is run through the given graph module. To implement a\n        pass, it is required to implement this function.\n\n        Args:\n            graph_module: The graph module we will run a pass on\n        '
        pass

    def requires(self, graph_module: GraphModule) -> None:
        if False:
            print('Hello World!')
        '\n        This function will be called before the pass is run and will check that\n        the given graph module contains the preconditions needed to run the\n        pass. It is not required to implement this function.\n\n        Args:\n            graph_module: The graph module we will run checks on\n        '
        pass

    def ensures(self, graph_module: GraphModule) -> None:
        if False:
            return 10
        '\n        This function will be called after the pass is run and will check that\n        the given graph module contains the postconditions needed to run the\n        pass. It is not required to implement this function.\n\n        Args:\n            graph_module: The graph module we will run checks on\n        '
        pass