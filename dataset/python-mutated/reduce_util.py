"""Utilities for reduce operations."""
import enum
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export

@tf_export('distribute.ReduceOp')
class ReduceOp(enum.Enum):
    """Indicates how a set of values should be reduced.

  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean ("average") of the values.
  """
    SUM = 'SUM'
    MEAN = 'MEAN'

    @staticmethod
    def from_variable_aggregation(aggregation):
        if False:
            print('Hello World!')
        mapping = {variable_scope.VariableAggregation.SUM: ReduceOp.SUM, variable_scope.VariableAggregation.MEAN: ReduceOp.MEAN}
        reduce_op = mapping.get(aggregation)
        if not reduce_op:
            raise ValueError('Could not convert from `tf.VariableAggregation` %s to`tf.distribute.ReduceOp` type' % aggregation)
        return reduce_op