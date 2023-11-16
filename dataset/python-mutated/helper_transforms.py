import collections
import itertools
import typing
import apache_beam as beam
from apache_beam import typehints
from apache_beam.internal.util import ArgumentPlaceholder
from apache_beam.transforms.combiners import _CurriedFn
from apache_beam.utils.windowed_value import WindowedValue

class LiftedCombinePerKey(beam.PTransform):
    """An implementation of CombinePerKey that does mapper-side pre-combining.
  """

    def __init__(self, combine_fn, args, kwargs):
        if False:
            while True:
                i = 10
        args_to_check = itertools.chain(args, kwargs.values())
        if isinstance(combine_fn, _CurriedFn):
            args_to_check = itertools.chain(args_to_check, combine_fn.args, combine_fn.kwargs.values())
        if any((isinstance(arg, ArgumentPlaceholder) for arg in args_to_check)):
            raise NotImplementedError('Deferred CombineFn side inputs.')
        self._combine_fn = beam.transforms.combiners.curry_combine_fn(combine_fn, args, kwargs)

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | beam.ParDo(PartialGroupByKeyCombiningValues(self._combine_fn)) | beam.GroupByKey() | beam.ParDo(FinishCombine(self._combine_fn))

class PartialGroupByKeyCombiningValues(beam.DoFn):
    """Aggregates values into a per-key-window cache.

  As bundles are in-memory-sized, we don't bother flushing until the very end.
  """

    def __init__(self, combine_fn):
        if False:
            print('Hello World!')
        self._combine_fn = combine_fn

    def setup(self):
        if False:
            while True:
                i = 10
        self._combine_fn.setup()

    def start_bundle(self):
        if False:
            while True:
                i = 10
        self._cache = collections.defaultdict(self._combine_fn.create_accumulator)

    def process(self, element, window=beam.DoFn.WindowParam):
        if False:
            while True:
                i = 10
        (k, vi) = element
        self._cache[k, window] = self._combine_fn.add_input(self._cache[k, window], vi)

    def finish_bundle(self):
        if False:
            i = 10
            return i + 15
        for ((k, w), va) in self._cache.items():
            yield WindowedValue((k, self._combine_fn.compact(va)), w.end, (w,))

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        self._combine_fn.teardown()

    def default_type_hints(self):
        if False:
            i = 10
            return i + 15
        hints = self._combine_fn.get_type_hints()
        K = typehints.TypeVariable('K')
        if hints.input_types:
            (args, kwargs) = hints.input_types
            args = (typehints.Tuple[K, args[0]],) + args[1:]
            hints = hints.with_input_types(*args, **kwargs)
        else:
            hints = hints.with_input_types(typehints.Tuple[K, typing.Any])
        hints = hints.with_output_types(typehints.Tuple[K, typing.Any])
        return hints

class FinishCombine(beam.DoFn):
    """Merges partially combined results.
  """

    def __init__(self, combine_fn):
        if False:
            while True:
                i = 10
        self._combine_fn = combine_fn

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self._combine_fn.setup()

    def process(self, element):
        if False:
            print('Hello World!')
        (k, vs) = element
        return [(k, self._combine_fn.extract_output(self._combine_fn.merge_accumulators(vs)))]

    def teardown(self):
        if False:
            return 10
        self._combine_fn.teardown()

    def default_type_hints(self):
        if False:
            return 10
        hints = self._combine_fn.get_type_hints()
        K = typehints.TypeVariable('K')
        hints = hints.with_input_types(typehints.Tuple[K, typing.Any])
        if hints.output_types:
            main_output_type = hints.simple_output_type('')
            hints = hints.with_output_types(typehints.Tuple[K, main_output_type])
        return hints