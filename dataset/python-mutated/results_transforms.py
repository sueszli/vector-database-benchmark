"""Provides a transform to drain PCollection in case of an error."""
from __future__ import annotations
import apache_beam as beam
import result
from typing import Any, Tuple

class DrainResultsOnError(beam.PTransform):
    """Transform that flushes an input PCollection if any error results
       are encountered.
    """

    def expand(self, objects: beam.PCollection[result.Result[Tuple[str, Any], Tuple[str, Exception]]]) -> beam.PCollection[result.Result[Tuple[str, Any], None]]:
        if False:
            print('Hello World!')
        'Count error results in collection and flush the input\n            in case of errors.\n\n        Args:\n            objects: PCollection. Sequence of Result objects.\n\n        Returns:\n            PCollection. Sequence of Result objects or empty PCollection.\n        '
        error_check = objects | 'Filter errors' >> beam.Filter(lambda result_item: result_item.is_err()) | 'Count number of errors' >> beam.combiners.Count.Globally() | 'Check if error count is zero' >> beam.Map(lambda x: x == 0)
        filtered_results = objects | 'Remove all results in case of errors' >> beam.Filter(lambda _, no_migration_error: bool(no_migration_error), no_migration_error=beam.pvalue.AsSingleton(error_check))
        return filtered_results