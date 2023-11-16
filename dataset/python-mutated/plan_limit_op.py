from typing import TYPE_CHECKING
from ray.data._internal.execution.operators.limit_operator import LimitOperator
if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.logical.operators.one_to_one_operator import Limit

def plan_limit_op(op: 'Limit', input_physical_dag: 'PhysicalOperator') -> 'PhysicalOperator':
    if False:
        i = 10
        return i + 15
    'Get the corresponding DAG of physical operators for Limit.\n\n    Note this method only converts the given `op`, but not its input dependencies.\n    See Planner.plan() for more details.\n    '
    return LimitOperator(op._limit, input_physical_dag)