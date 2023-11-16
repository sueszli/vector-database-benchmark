from ..Node import Node
from .common import CtrlNode

class UniOpNode(Node):
    """Generic node for performing any operation like Out = In.fn()"""

    def __init__(self, name, fn):
        if False:
            for i in range(10):
                print('nop')
        self.fn = fn
        Node.__init__(self, name, terminals={'In': {'io': 'in'}, 'Out': {'io': 'out', 'bypass': 'In'}})

    def process(self, **args):
        if False:
            while True:
                i = 10
        return {'Out': getattr(args['In'], self.fn)()}

class BinOpNode(CtrlNode):
    """Generic node for performing any operation like A.fn(B)"""
    _dtypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']
    uiTemplate = [('outputType', 'combo', {'values': ['no change', 'input A', 'input B'] + _dtypes, 'index': 0})]

    def __init__(self, name, fn):
        if False:
            for i in range(10):
                print('nop')
        self.fn = fn
        CtrlNode.__init__(self, name, terminals={'A': {'io': 'in'}, 'B': {'io': 'in'}, 'Out': {'io': 'out', 'bypass': 'A'}})

    def process(self, **args):
        if False:
            while True:
                i = 10
        if isinstance(self.fn, tuple):
            for name in self.fn:
                try:
                    fn = getattr(args['A'], name)
                    break
                except AttributeError as e:
                    pass
            else:
                raise e
        else:
            fn = getattr(args['A'], self.fn)
        out = fn(args['B'])
        if out is NotImplemented:
            raise Exception('Operation %s not implemented between %s and %s' % (fn, str(type(args['A'])), str(type(args['B']))))
        typ = self.stateGroup.state()['outputType']
        if typ == 'no change':
            pass
        elif typ == 'input A':
            out = out.astype(args['A'].dtype)
        elif typ == 'input B':
            out = out.astype(args['B'].dtype)
        else:
            out = out.astype(typ)
        return {'Out': out}

class AbsNode(UniOpNode):
    """Returns abs(Inp). Does not check input types."""
    nodeName = 'Abs'

    def __init__(self, name):
        if False:
            while True:
                i = 10
        UniOpNode.__init__(self, name, '__abs__')

class AddNode(BinOpNode):
    """Returns A + B. Does not check input types."""
    nodeName = 'Add'

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        BinOpNode.__init__(self, name, '__add__')

class SubtractNode(BinOpNode):
    """Returns A - B. Does not check input types."""
    nodeName = 'Subtract'

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        BinOpNode.__init__(self, name, '__sub__')

class MultiplyNode(BinOpNode):
    """Returns A * B. Does not check input types."""
    nodeName = 'Multiply'

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        BinOpNode.__init__(self, name, '__mul__')

class DivideNode(BinOpNode):
    """Returns A / B. Does not check input types."""
    nodeName = 'Divide'

    def __init__(self, name):
        if False:
            return 10
        BinOpNode.__init__(self, name, ('__truediv__', '__div__'))

class FloorDivideNode(BinOpNode):
    """Returns A // B. Does not check input types."""
    nodeName = 'FloorDivide'

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        BinOpNode.__init__(self, name, '__floordiv__')