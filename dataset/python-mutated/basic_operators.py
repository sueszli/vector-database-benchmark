from ryven.node_env import *
guis = import_guis(__file__)

class OperatorNodeBase(Node):
    """
    Base class for nodes implementing a binary operation.
    """
    version = 'v0.2'
    init_inputs = [NodeInputType(), NodeInputType()]
    init_outputs = [NodeOutputType()]
    GUI = guis.OperatorNodeBaseGui

    def __init__(self, params):
        if False:
            print('Hello World!')
        super().__init__(params)
        self.num_inputs = 0

    def place_event(self):
        if False:
            while True:
                i = 10
        self.num_inputs = len(self.inputs)

    def add_op_inp(self):
        if False:
            i = 10
            return i + 15
        self.create_input()
        self.num_inputs += 1

    def remove_op_input(self, index):
        if False:
            return 10
        self.delete_input(index)
        self.num_inputs -= 1

    def update_event(self, inp=-1):
        if False:
            return 10
        self.set_output_val(0, Data(self.apply_op([self.input(i).payload for i in range(len(self.inputs))])))

    def apply_op(self, elements: list):
        if False:
            i = 10
            return i + 15
        return None
'\n    logical operators\n'

class LogicNodeBase(OperatorNodeBase):
    GUI = guis.LogicNodeBaseGui

class NOT_Node(LogicNodeBase):
    title = 'not'

    def apply_op(self, elements: list):
        if False:
            for i in range(10):
                print('nop')
        return all([not bool(e) for e in elements])

class AND_Node(LogicNodeBase):
    title = 'and'

    def apply_op(self, elements: list):
        if False:
            while True:
                i = 10
        return all(elements)

class NAND_Node(LogicNodeBase):
    title = 'nand'

    def apply_op(self, elements: list):
        if False:
            return 10
        return not all(elements)

class OR_Node(LogicNodeBase):
    title = 'or'

    def apply_op(self, elements: list):
        if False:
            return 10
        return any(elements)

class NOR_Node(LogicNodeBase):
    title = 'nor'

    def apply_op(self, elements: list):
        if False:
            while True:
                i = 10
        return not any(elements)

class XOR_Node(LogicNodeBase):
    title = 'xor'

    def apply_op(self, elements: list):
        if False:
            i = 10
            return i + 15
        return len(list(filter(lambda x: bool(x), elements))) % 2 != 0

class XNOR_Node(LogicNodeBase):
    title = 'xnor'

    def apply_op(self, elements: list):
        if False:
            i = 10
            return i + 15
        return len(list(filter(lambda x: bool(x), elements))) % 2 == 0
logic_nodes = [NOT_Node, AND_Node, NAND_Node, OR_Node, NOR_Node, XOR_Node, XNOR_Node]
'\n    arithmetic operators\n'

class ArithmeticNodeBase(OperatorNodeBase):
    GUI = guis.ArithNodeBaseGui

class Plus_Node(ArithmeticNodeBase):
    title = '+'

    def apply_op(self, elements: list):
        if False:
            return 10
        v = elements[0]
        for e in elements[1:]:
            v = v + e
        return v

class Minus_Node(ArithmeticNodeBase):
    title = '-'

    def apply_op(self, elements: list):
        if False:
            i = 10
            return i + 15
        v = elements[0]
        for e in elements[1:]:
            v = v - e
        return v

class Multiply_Node(ArithmeticNodeBase):
    title = '*'

    def apply_op(self, elements: list):
        if False:
            while True:
                i = 10
        v = elements[0]
        for e in elements[1:]:
            v *= e
        return v

class Divide_Node(ArithmeticNodeBase):
    title = '/'

    def apply_op(self, elements: list):
        if False:
            for i in range(10):
                print('nop')
        v = elements[0]
        for e in elements[1:]:
            v = v / e
        return v

class Power_Node(ArithmeticNodeBase):
    title = '**'

    def apply_op(self, elements: list):
        if False:
            return 10
        v = elements[0]
        for e in elements[1:]:
            v = v ** e
        return v
arithmetic_nodes = [Plus_Node, Minus_Node, Multiply_Node, Divide_Node, Power_Node]
'\n    comparison operators\n'

class ComparatorNodeBase(OperatorNodeBase):
    GUI = guis.CompNodeBaseGui

    def apply_op(self, elements: list):
        if False:
            print('Hello World!')
        b = True
        for i in range(1, len(elements)):
            b = b and self.comp(elements[i - 1], elements[i])
        return b

    def comp(self, a, b) -> bool:
        if False:
            while True:
                i = 10
        return False

class Equal_Node(ComparatorNodeBase):
    title = '=='

    def comp(self, a, b) -> bool:
        if False:
            while True:
                i = 10
        return a == b

class NotEqual_Node(ComparatorNodeBase):
    title = '!='

    def comp(self, a, b) -> bool:
        if False:
            i = 10
            return i + 15
        return a != b

class Greater_Node(ComparatorNodeBase):
    title = '>'

    def comp(self, a, b) -> bool:
        if False:
            return 10
        return a > b

class GreaterEq_Node(ComparatorNodeBase):
    title = '>='

    def comp(self, a, b) -> bool:
        if False:
            while True:
                i = 10
        return a >= b

class Less_Node(ComparatorNodeBase):
    title = '<'

    def comp(self, a, b) -> bool:
        if False:
            return 10
        return a < b

class LessEq_Node(ComparatorNodeBase):
    title = '<='

    def comp(self, a, b) -> bool:
        if False:
            print('Hello World!')
        return a <= b
comparator_nodes = [Equal_Node, NotEqual_Node, Greater_Node, GreaterEq_Node, Less_Node, LessEq_Node]
'\n    export\n'
nodes = [*logic_nodes, *arithmetic_nodes, *comparator_nodes]