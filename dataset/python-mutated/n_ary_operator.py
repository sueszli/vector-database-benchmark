from ray.data._internal.logical.interfaces import LogicalOperator

class NAry(LogicalOperator):
    """Base class for n-ary operators, which take multiple input operators."""

    def __init__(self, *input_ops: LogicalOperator):
        if False:
            print('Hello World!')
        '\n        Args:\n            input_ops: The input operators.\n        '
        super().__init__(self.__class__.__name__, list(input_ops))

class Zip(NAry):
    """Logical operator for zip."""

    def __init__(self, left_input_op: LogicalOperator, right_input_op: LogicalOperator):
        if False:
            print('Hello World!')
        '\n        Args:\n            left_input_ops: The input operator at left hand side.\n            right_input_op: The input operator at right hand side.\n        '
        super().__init__(left_input_op, right_input_op)

class Union(NAry):
    """Logical operator for union."""

    def __init__(self, *input_ops: LogicalOperator):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*input_ops)