from typing import *
import apache_beam as beam
from apache_beam.dataframe import convert
from apache_beam.dataframe import expressions

class ExpressionCache(object):
    """Utility class for caching deferred DataFrames expressions.

  This is cache is currently a light-weight wrapper around the
  TO_PCOLLECTION_CACHE in the beam.dataframes.convert module and the
  computed_pcollections in the interactive module.

  Example::

    df : beam.dataframe.DeferredDataFrame = ...
    ...
    cache = ExpressionCache()
    cache.replace_with_cached(df._expr)

  This will automatically link the instance to the existing caches. After it is
  created, the cache can then be used to modify an existing deferred dataframe
  expression tree to replace nodes with computed PCollections.

  This object can be created and destroyed whenever. This class holds no state
  and the only side-effect is modifying the given expression.
  """

    def __init__(self, pcollection_cache=None, computed_cache=None):
        if False:
            print('Hello World!')
        from apache_beam.runners.interactive import interactive_environment as ie
        self._pcollection_cache = convert.TO_PCOLLECTION_CACHE if pcollection_cache is None else pcollection_cache
        self._computed_cache = ie.current_env().computed_pcollections if computed_cache is None else computed_cache

    def replace_with_cached(self, expr: expressions.Expression) -> Dict[str, expressions.Expression]:
        if False:
            while True:
                i = 10
        'Replaces any previously computed expressions with PlaceholderExpressions.\n\n    This is used to correctly read any expressions that were cached in previous\n    runs. This enables the InteractiveRunner to prune off old calculations from\n    the expression tree.\n    '
        replaced_inputs: Dict[str, expressions.Expression] = {}
        self._replace_with_cached_recur(expr, replaced_inputs)
        return replaced_inputs

    def _replace_with_cached_recur(self, expr: expressions.Expression, replaced_inputs: Dict[str, expressions.Expression]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Recursive call for `replace_with_cached`.\n\n    Recurses through the expression tree and replaces any cached inputs with\n    `PlaceholderExpression`s.\n    '
        final_inputs = []
        for input in expr.args():
            pc = self._get_cached(input)
            if self._is_computed(pc):
                if input._id in replaced_inputs:
                    cached = replaced_inputs[input._id]
                else:
                    cached = expressions.PlaceholderExpression(input.proxy(), self._pcollection_cache[input._id])
                    replaced_inputs[input._id] = cached
                final_inputs.append(cached)
            else:
                final_inputs.append(input)
                self._replace_with_cached_recur(input, replaced_inputs)
        expr._args = tuple(final_inputs)

    def _get_cached(self, expr: expressions.Expression) -> Optional[beam.PCollection]:
        if False:
            print('Hello World!')
        'Returns the PCollection associated with the expression.'
        return self._pcollection_cache.get(expr._id, None)

    def _is_computed(self, pc: beam.PCollection) -> bool:
        if False:
            while True:
                i = 10
        'Returns True if the PCollection has been run and computed.'
        return pc is not None and pc in self._computed_cache