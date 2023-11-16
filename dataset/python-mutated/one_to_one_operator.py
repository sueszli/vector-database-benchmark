import abc
from typing import Optional
from ray.data._internal.logical.interfaces import LogicalOperator

class AbstractOneToOne(LogicalOperator):
    """Abstract class for one-to-one logical operators, which
    have one input and one output dependency.
    """

    def __init__(self, name: str, input_op: Optional[LogicalOperator]):
        if False:
            print('Hello World!')
        '\n        Args:\n            name: Name for this operator. This is the name that will appear when\n                inspecting the logical plan of a Dataset.\n            input_op: The operator preceding this operator in the plan DAG. The outputs\n                of `input_op` will be the inputs to this operator.\n        '
        super().__init__(name, [input_op] if input_op else [])

    @property
    def input_dependency(self) -> LogicalOperator:
        if False:
            while True:
                i = 10
        return self._input_dependencies[0]

    @property
    @abc.abstractmethod
    def can_modify_num_rows(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether this operator can modify the number of rows,\n        i.e. number of input rows != number of output rows.'

class Limit(AbstractOneToOne):
    """Logical operator for limit."""

    def __init__(self, input_op: LogicalOperator, limit: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(f'limit={limit}', input_op)
        self._limit = limit

    @property
    def can_modify_num_rows(self) -> bool:
        if False:
            return 10
        return True