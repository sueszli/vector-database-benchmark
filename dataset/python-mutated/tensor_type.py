from torch.fx.experimental.unification import Var
from ._compatibility import compatibility

@compatibility(is_backward_compatible=False)
class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    def __init__(self, dim):
        if False:
            while True:
                i = 10
        self.__origin__ = TensorType
        self.__args__ = dim

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'TensorType[{self.__args__}]'

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, self.__class__):
            return list(self.__args__) == list(other.__args__)
        else:
            return False

    @staticmethod
    def __class_getitem__(*args):
        if False:
            return 10
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))

class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__name__ = '_DynType'

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, self.__class__)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'Dyn'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Dyn'
Dyn = _DynType()

@compatibility(is_backward_compatible=False)
def is_consistent(t1, t2):
    if False:
        i = 10
        return i + 15
    '\n    A binary relation denoted by ~ that determines if t1 is consistent with t2.\n    The relation is reflexive, symmetric but not transitive.\n    returns True if t1 and t2 are consistent and False otherwise.\n    Example:\n        Dyn ~ TensorType((1,2,3))\n        int ~ Dyn\n        int ~ int\n        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))\n    '
    if t1 == t2:
        return True
    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return True
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and all((is_consistent(elem1, elem2) for (elem1, elem2) in zip(t1.__args__, t2.__args__)))
    else:
        return False

@compatibility(is_backward_compatible=False)
def is_more_precise(t1, t2):
    if False:
        print('Hello World!')
    '\n    A binary relation denoted by <= that determines if t1 is more precise than t2.\n    The relation is reflexive and transitive.\n    returns True if t1 is more precise than t2 and False otherwise.\n    Example:\n        Dyn >= TensorType((1,2,3))\n        int >= Dyn\n        int >= int\n        TensorType((1,Dyn,3)) <= TensorType((1,2,3))\n    '
    if t1 == t2:
        return True
    if isinstance(t2, _DynType):
        return True
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and all((is_more_precise(elem1, elem2) for (elem1, elem2) in zip(t1.__args__, t2.__args__)))
    else:
        return False