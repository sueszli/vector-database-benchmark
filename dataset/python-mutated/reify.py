"""Module for transforms that reifies and unreifies PCollection values with
window info.

For internal use only; no backwards-compatibility guarantees.
"""
from typing import Optional
import apache_beam as beam
from apache_beam.runners.interactive import cache_manager as cache
from apache_beam.testing import test_stream
from apache_beam.transforms.window import WindowedValue
READ_CACHE = 'ReadCache_'
WRITE_CACHE = 'WriteCache_'

class Reify(beam.DoFn):
    """Reifies elements with window info into windowed values.

  Internally used to capture window info with each element into cache for
  replayability.
  """

    def process(self, e, w=beam.DoFn.WindowParam, p=beam.DoFn.PaneInfoParam, t=beam.DoFn.TimestampParam):
        if False:
            return 10
        yield test_stream.WindowedValueHolder(WindowedValue(e, t, [w], p))

class Unreify(beam.DoFn):
    """Unreifies elements from windowed values.

  Cached values are elements with window info. This unpacks the elements.
  """

    def process(self, e):
        if False:
            i = 10
            return i + 15
        if isinstance(e, beam.Row) and hasattr(e, 'windowed_value'):
            yield e.windowed_value

def reify_to_cache(pcoll: beam.pvalue.PCollection, cache_key: str, cache_manager: cache.CacheManager, reify_label: Optional[str]=None, write_cache_label: Optional[str]=None, is_capture: bool=False) -> beam.pvalue.PValue:
    if False:
        while True:
            i = 10
    'Reifies elements into windowed values and write to cache.\n\n  Args:\n    pcoll: The PCollection to be cached.\n    cache_key: The key of the cache.\n    cache_manager: The cache manager to manage the cache.\n    reify_label: (optional) A transform label for the Reify transform.\n    write_cache_label: (optional) A transform label for the cache-writing\n      transform.\n    is_capture: Whether the cache is capturing a record of recordable sources.\n  '
    if not reify_label:
        reify_label = '{}{}{}'.format('ReifyBefore_', WRITE_CACHE, cache_key)
    if not write_cache_label:
        write_cache_label = '{}{}'.format(WRITE_CACHE, cache_key)
    return pcoll | reify_label >> beam.ParDo(Reify()) | write_cache_label >> cache.WriteCache(cache_manager, cache_key, is_capture=is_capture)

def unreify_from_cache(pipeline: beam.Pipeline, cache_key: str, cache_manager: cache.CacheManager, element_type: Optional[type]=None, source_label: Optional[str]=None, unreify_label: Optional[str]=None) -> beam.pvalue.PCollection:
    if False:
        while True:
            i = 10
    "Reads from cache and unreifies elements from windowed values.\n\n  pipeline: The pipeline that's reading from the cache.\n  cache_key: The key of the cache.\n  cache_manager: The cache manager to manage the cache.\n  element_type: (optional) The element type of the PCollection's elements.\n  source_label: (optional) A transform label for the cache-reading transform.\n  unreify_label: (optional) A transform label for the Unreify transform.\n  "
    if not source_label:
        source_label = '{}{}'.format(READ_CACHE, cache_key)
    if not unreify_label:
        unreify_label = '{}{}{}'.format('UnreifyAfter_', READ_CACHE, cache_key)
    read_cache = pipeline | source_label >> cache.ReadCache(cache_manager, cache_key)
    if element_type:
        return read_cache | unreify_label >> beam.ParDo(Unreify()).with_output_types(element_type)
    return read_cache | unreify_label >> beam.ParDo(Unreify())