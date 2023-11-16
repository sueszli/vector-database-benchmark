"""Module to instrument interactivity to the given pipeline.

For internal use only; no backwards-compatibility guarantees.
This module accesses current interactive environment and analyzes given pipeline
to transform original pipeline into a one-shot pipeline with interactivity.
"""
import logging
from typing import Dict
import apache_beam as beam
from apache_beam.pipeline import PipelineVisitor
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import pipeline_fragment as pf
from apache_beam.runners.interactive import background_caching_job
from apache_beam.runners.interactive import utils
from apache_beam.runners.interactive.caching.cacheable import Cacheable
from apache_beam.runners.interactive.caching.cacheable import CacheKey
from apache_beam.runners.interactive.caching.reify import WRITE_CACHE
from apache_beam.runners.interactive.caching.reify import reify_to_cache
from apache_beam.runners.interactive.caching.reify import unreify_from_cache
from apache_beam.testing import test_stream
_LOGGER = logging.getLogger(__name__)

class PipelineInstrument(object):
    """A pipeline instrument for pipeline to be executed by interactive runner.

  This module should never depend on underlying runner that interactive runner
  delegates. It instruments the original instance of pipeline directly by
  appending or replacing transforms with help of cache. It provides
  interfaces to recover states of original pipeline. It's the interactive
  runner's responsibility to coordinate supported underlying runners to run
  the pipeline instrumented and recover the original pipeline states if needed.
  """

    def __init__(self, pipeline, options=None):
        if False:
            while True:
                i = 10
        self._pipeline = pipeline
        self._user_pipeline = ie.current_env().user_pipeline(pipeline)
        if not self._user_pipeline:
            self._user_pipeline = pipeline
        self._cache_manager = ie.current_env().get_cache_manager(self._user_pipeline, create_if_absent=True)
        if background_caching_job.has_source_to_cache(self._user_pipeline):
            self._cache_manager = ie.current_env().get_cache_manager(self._user_pipeline)
        self._background_caching_pipeline = beam.pipeline.Pipeline.from_runner_api(pipeline.to_runner_api(), pipeline.runner, options)
        ie.current_env().add_derived_pipeline(self._pipeline, self._background_caching_pipeline)
        (self._original_pipeline_proto, context) = self._pipeline.to_runner_api(return_context=True)
        self._unbounded_sources = utils.unbounded_sources(self._background_caching_pipeline)
        self._pcoll_to_pcoll_id = pcoll_to_pcoll_id(self._pipeline, context)
        self._cacheables = self.find_cacheables()
        self._cached_pcoll_read = {}
        self._runner_pcoll_to_user_pcoll = {}
        self._pruned_pipeline_proto = None
        self._extended_targets = set()
        self._ignored_targets = set()
        self.cached_pcolls = set()

    def instrumented_pipeline_proto(self):
        if False:
            print('Hello World!')
        'Always returns a new instance of portable instrumented proto.'
        targets = set(self._runner_pcoll_to_user_pcoll.keys())
        targets.update(self._extended_targets)
        targets = targets.difference(self._ignored_targets)
        if len(targets) > 0:
            return pf.PipelineFragment(list(targets)).deduce_fragment().to_runner_api()
        return self._pipeline.to_runner_api()

    def _required_components(self, pipeline_proto, required_transforms_ids, visited, follow_outputs=False, follow_inputs=False):
        if False:
            i = 10
            return i + 15
        'Returns the components and subcomponents of the given transforms.\n\n    This method returns required components such as transforms and PCollections\n    related to the given transforms and to all of their subtransforms. This\n    method accomplishes this recursively.\n    '
        if not required_transforms_ids:
            return ({}, {})
        transforms = pipeline_proto.components.transforms
        pcollections = pipeline_proto.components.pcollections
        required_transforms = {k: transforms[k] for k in required_transforms_ids}
        pcollection_ids = [pc for t in required_transforms.values() for pc in t.outputs.values()]
        required_pcollections = {pc_id: pcollections[pc_id] for pc_id in pcollection_ids}
        subtransforms = {}
        subpcollections = {}
        for (transform_id, transform) in required_transforms.items():
            if transform_id in pipeline_proto.root_transform_ids:
                continue
            (t, pc) = self._required_components(pipeline_proto, transform.subtransforms, visited, follow_outputs=False, follow_inputs=False)
            subtransforms.update(t)
            subpcollections.update(pc)
        if follow_outputs:
            outputs = [pc_id for t in required_transforms.values() for pc_id in t.outputs.values()]
            visited_copy = visited.copy()
            consuming_transforms = {t_id: t for (t_id, t) in transforms.items() if set(outputs).intersection(set(t.inputs.values()))}
            consuming_transforms = set(consuming_transforms.keys())
            visited.update(consuming_transforms)
            consuming_transforms = consuming_transforms - visited_copy
            (t, pc) = self._required_components(pipeline_proto, list(consuming_transforms), visited, follow_outputs, follow_inputs)
            subtransforms.update(t)
            subpcollections.update(pc)
        if follow_inputs:
            inputs = [pc_id for t in required_transforms.values() for pc_id in t.inputs.values()]
            producing_transforms = {t_id: t for (t_id, t) in transforms.items() if set(inputs).intersection(set(t.outputs.values()))}
            (t, pc) = self._required_components(pipeline_proto, list(producing_transforms.keys()), visited, follow_outputs, follow_inputs)
            subtransforms.update(t)
            subpcollections.update(pc)
        required_transforms.update(subtransforms)
        required_pcollections.update(subpcollections)
        return (required_transforms, required_pcollections)

    def prune_subgraph_for(self, pipeline, required_transform_ids):
        if False:
            print('Hello World!')
        (pipeline_proto, context) = pipeline.to_runner_api(return_context=True)
        roots = [root for root in pipeline_proto.root_transform_ids]
        (t, p) = self._required_components(pipeline_proto, roots + required_transform_ids, set(), follow_outputs=True, follow_inputs=True)

        def set_proto_map(proto_map, new_value):
            if False:
                print('Hello World!')
            proto_map.clear()
            for (key, value) in new_value.items():
                proto_map[key].CopyFrom(value)
        pipeline_to_execute = beam_runner_api_pb2.Pipeline()
        pipeline_to_execute.root_transform_ids[:] = roots
        set_proto_map(pipeline_to_execute.components.transforms, t)
        set_proto_map(pipeline_to_execute.components.pcollections, p)
        set_proto_map(pipeline_to_execute.components.coders, context.to_runner_api().coders)
        set_proto_map(pipeline_to_execute.components.windowing_strategies, context.to_runner_api().windowing_strategies)
        for root_id in roots:
            root = pipeline_to_execute.components.transforms[root_id]
            root.subtransforms[:] = [transform_id for transform_id in root.subtransforms if transform_id in pipeline_to_execute.components.transforms]
        return pipeline_to_execute

    def background_caching_pipeline_proto(self):
        if False:
            while True:
                i = 10
        'Returns the background caching pipeline.\n\n    This method creates a background caching pipeline by: adding writes to cache\n    from each unbounded source (done in the instrument method), and cutting out\n    all components (transform, PCollections, coders, windowing strategies) that\n    are not the unbounded sources or writes to cache (or subtransforms thereof).\n    '
        (pipeline_proto, context) = self._background_caching_pipeline.to_runner_api(return_context=True)
        sources = utils.unbounded_sources(self._background_caching_pipeline)
        roots = [root for root in pipeline_proto.root_transform_ids]
        transforms = pipeline_proto.components.transforms
        caching_transform_ids = [t_id for root in roots for t_id in transforms[root].subtransforms if WRITE_CACHE in t_id]
        required_transform_labels = [src.full_label for src in sources]
        unbounded_source_ids = [k for (k, v) in transforms.items() if v.unique_name in required_transform_labels]
        required_transform_ids = roots + caching_transform_ids + unbounded_source_ids
        (t, p) = self._required_components(pipeline_proto, required_transform_ids, set())

        def set_proto_map(proto_map, new_value):
            if False:
                while True:
                    i = 10
            proto_map.clear()
            for (key, value) in new_value.items():
                proto_map[key].CopyFrom(value)
        pipeline_to_execute = beam_runner_api_pb2.Pipeline()
        pipeline_to_execute.root_transform_ids[:] = roots
        set_proto_map(pipeline_to_execute.components.transforms, t)
        set_proto_map(pipeline_to_execute.components.pcollections, p)
        set_proto_map(pipeline_to_execute.components.coders, context.to_runner_api().coders)
        set_proto_map(pipeline_to_execute.components.windowing_strategies, context.to_runner_api().windowing_strategies)
        for root_id in roots:
            root = pipeline_to_execute.components.transforms[root_id]
            root.subtransforms[:] = [transform_id for transform_id in root.subtransforms if transform_id in pipeline_to_execute.components.transforms]
        return pipeline_to_execute

    @property
    def cacheables(self) -> Dict[str, Cacheable]:
        if False:
            while True:
                i = 10
        "Returns the Cacheables by PCollection ids.\n\n    If you're already working with user defined pipelines and PCollections,\n    do not build a PipelineInstrument just to get the cacheables. Instead,\n    use apache_beam.runners.interactive.utils.cacheables.\n    "
        return self._cacheables

    @property
    def has_unbounded_sources(self):
        if False:
            return 10
        'Returns whether the pipeline has any recordable sources.\n    '
        return len(self._unbounded_sources) > 0

    @property
    def original_pipeline_proto(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a snapshot of the pipeline proto before instrumentation.'
        return self._original_pipeline_proto

    @property
    def user_pipeline(self):
        if False:
            return 10
        'Returns a reference to the pipeline instance defined by the user. If a\n    pipeline has no cacheable PCollection and the user pipeline cannot be\n    found, return None indicating there is nothing to be cached in the user\n    pipeline.\n\n    The pipeline given for instrumenting and mutated in this class is not\n    necessarily the pipeline instance defined by the user. From the watched\n    scopes, this class figures out what the user pipeline instance is.\n    This metadata can be used for tracking pipeline results.\n    '
        return self._user_pipeline

    @property
    def runner_pcoll_to_user_pcoll(self):
        if False:
            print('Hello World!')
        'Returns cacheable PCollections correlated from instances in the runner\n    pipeline to instances in the user pipeline.'
        return self._runner_pcoll_to_user_pcoll

    def find_cacheables(self) -> Dict[str, Cacheable]:
        if False:
            i = 10
            return i + 15
        'Finds PCollections that need to be cached for analyzed pipeline.\n\n    There might be multiple pipelines defined and watched, this will only find\n    cacheables belong to the analyzed pipeline.\n    '
        result = {}
        cacheables = utils.cacheables()
        for (_, cacheable) in cacheables.items():
            if cacheable.pcoll.pipeline is not self._user_pipeline:
                continue
            pcoll_id = self.pcoll_id(cacheable.pcoll)
            if not pcoll_id:
                _LOGGER.debug('Unable to retrieve PCollection id for %s. Ignored.', cacheable.pcoll)
                continue
            result[self.pcoll_id(cacheable.pcoll)] = cacheable
        return result

    def instrument(self):
        if False:
            return 10
        "Instruments original pipeline with cache.\n\n    For cacheable output PCollection, if cache for the key doesn't exist, do\n    _write_cache(); for cacheable input PCollection, if cache for the key\n    exists, do _read_cache(). No instrument in any other situation.\n\n    Modifies:\n      self._pipeline\n    "
        cacheable_inputs = set()
        all_inputs = set()
        all_outputs = set()
        unbounded_source_pcolls = set()

        class InstrumentVisitor(PipelineVisitor):
            """Visitor utilizes cache to instrument the pipeline."""

            def __init__(self, pin):
                if False:
                    i = 10
                    return i + 15
                self._pin = pin

            def enter_composite_transform(self, transform_node):
                if False:
                    print('Hello World!')
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    while True:
                        i = 10
                if isinstance(transform_node.transform, tuple(ie.current_env().options.recordable_sources)):
                    unbounded_source_pcolls.update(transform_node.outputs.values())
                cacheable_inputs.update(self._pin._cacheable_inputs(transform_node))
                (ins, outs) = self._pin._all_inputs_outputs(transform_node)
                all_inputs.update(ins)
                all_outputs.update(outs)
        v = InstrumentVisitor(self)
        self._pipeline.visit(v)
        self._extended_targets.update(all_outputs.difference(all_inputs))
        cacheable_inputs.update(unbounded_source_pcolls)
        for cacheable_input in cacheable_inputs:
            self._read_cache(self._pipeline, cacheable_input, cacheable_input in unbounded_source_pcolls)
        self._replace_with_cached_inputs(self._pipeline)
        for (_, cacheable) in self._cacheables.items():
            self._write_cache(self._pipeline, cacheable.pcoll, ignore_unbounded_reads=True)
        if self.has_unbounded_sources:
            for source in self._unbounded_sources:
                self._write_cache(self._background_caching_pipeline, source.outputs[None], output_as_extended_target=False, is_capture=True)

            class TestStreamVisitor(PipelineVisitor):

                def __init__(self):
                    if False:
                        print('Hello World!')
                    self.test_stream = None

                def enter_composite_transform(self, transform_node):
                    if False:
                        i = 10
                        return i + 15
                    self.visit_transform(transform_node)

                def visit_transform(self, transform_node):
                    if False:
                        while True:
                            i = 10
                    if self.test_stream is None and isinstance(transform_node.transform, test_stream.TestStream):
                        self.test_stream = transform_node.full_label
            v = TestStreamVisitor()
            self._pipeline.visit(v)
            pipeline_proto = self._pipeline.to_runner_api(return_context=False)
            test_stream_id = ''
            for (t_id, t) in pipeline_proto.components.transforms.items():
                if t.unique_name == v.test_stream:
                    test_stream_id = t_id
                    break
            self._pruned_pipeline_proto = self.prune_subgraph_for(self._pipeline, [test_stream_id])
            pruned_pipeline = beam.Pipeline.from_runner_api(proto=self._pruned_pipeline_proto, runner=self._pipeline.runner, options=self._pipeline._options)
            ie.current_env().add_derived_pipeline(self._pipeline, pruned_pipeline)
            self._pipeline = pruned_pipeline

    def preprocess(self):
        if False:
            return 10
        "Pre-processes the pipeline.\n\n    Since the pipeline instance in the class might not be the same instance\n    defined in the user code, the pre-process will figure out the relationship\n    of cacheable PCollections between these 2 instances by replacing 'pcoll'\n    fields in the cacheable dictionary with ones from the running instance.\n    "

        class PreprocessVisitor(PipelineVisitor):

            def __init__(self, pin):
                if False:
                    for i in range(10):
                        print('nop')
                self._pin = pin

            def enter_composite_transform(self, transform_node):
                if False:
                    return 10
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    print('Hello World!')
                for in_pcoll in transform_node.inputs:
                    self._process(in_pcoll)
                for out_pcoll in transform_node.outputs.values():
                    self._process(out_pcoll)

            def _process(self, pcoll):
                if False:
                    for i in range(10):
                        print('nop')
                pcoll_id = self._pin._pcoll_to_pcoll_id.get(str(pcoll), '')
                if pcoll_id in self._pin._cacheables:
                    pcoll_id = self._pin.pcoll_id(pcoll)
                    user_pcoll = self._pin._cacheables[pcoll_id].pcoll
                    if pcoll_id in self._pin._cacheables and user_pcoll != pcoll:
                        self._pin._runner_pcoll_to_user_pcoll[pcoll] = user_pcoll
                        self._pin._cacheables[pcoll_id].pcoll = pcoll
        v = PreprocessVisitor(self)
        self._pipeline.visit(v)

    def _write_cache(self, pipeline, pcoll, output_as_extended_target=True, ignore_unbounded_reads=False, is_capture=False):
        if False:
            print('Hello World!')
        'Caches a cacheable PCollection.\n\n    For the given PCollection, by appending sub transform part that materialize\n    the PCollection through sink into cache implementation. The cache write is\n    not immediate. It happens when the runner runs the transformed pipeline\n    and thus not usable for this run as intended. This function always writes\n    the cache for the given PCollection as long as the PCollection belongs to\n    the pipeline being instrumented and the keyed cache is absent.\n\n    Modifies:\n      pipeline\n    '
        if pcoll.pipeline is not pipeline:
            return
        if ignore_unbounded_reads:
            ignore = False
            producer = pcoll.producer
            while producer:
                if isinstance(producer.transform, tuple(ie.current_env().options.recordable_sources)):
                    ignore = True
                    break
                producer = producer.parent
            if ignore:
                self._ignored_targets.add(pcoll)
                return
        key = self.cache_key(pcoll)
        if not self._cache_manager.exists('full', key):
            self.cached_pcolls.add(self.runner_pcoll_to_user_pcoll.get(pcoll, pcoll))
            extended_target = reify_to_cache(pcoll=pcoll, cache_key=key, cache_manager=self._cache_manager, is_capture=is_capture)
            if output_as_extended_target:
                self._extended_targets.add(extended_target)

    def _read_cache(self, pipeline, pcoll, is_unbounded_source_output):
        if False:
            return 10
        'Reads a cached pvalue.\n\n    A noop will cause the pipeline to execute the transform as\n    it is and cache nothing from this transform for next run.\n\n    Modifies:\n      pipeline\n    '
        if pcoll.pipeline is not pipeline:
            return
        key = self.cache_key(pcoll)
        is_cached = self._cache_manager.exists('full', key)
        is_computed = pcoll in self._runner_pcoll_to_user_pcoll and self._runner_pcoll_to_user_pcoll[pcoll] in ie.current_env().computed_pcollections
        if is_cached and is_computed or is_unbounded_source_output:
            if key not in self._cached_pcoll_read:
                pcoll_from_cache = unreify_from_cache(pipeline=pipeline, cache_key=key, cache_manager=self._cache_manager)
                self._cached_pcoll_read[key] = pcoll_from_cache

    def _replace_with_cached_inputs(self, pipeline):
        if False:
            print('Hello World!')
        'Replace PCollection inputs in the pipeline with cache if possible.\n\n    For any input PCollection, find out whether there is valid cache. If so,\n    replace the input of the AppliedPTransform with output of the\n    AppliedPtransform that sources pvalue from the cache. If there is no valid\n    cache, noop.\n    '
        if self.has_unbounded_sources:

            class CacheableUnboundedPCollectionVisitor(PipelineVisitor):

                def __init__(self, pin):
                    if False:
                        print('Hello World!')
                    self._pin = pin
                    self.unbounded_pcolls = set()

                def enter_composite_transform(self, transform_node):
                    if False:
                        i = 10
                        return i + 15
                    self.visit_transform(transform_node)

                def visit_transform(self, transform_node):
                    if False:
                        i = 10
                        return i + 15
                    if transform_node.outputs:
                        for output_pcoll in transform_node.outputs.values():
                            key = self._pin.cache_key(output_pcoll)
                            if key in self._pin._cached_pcoll_read:
                                self.unbounded_pcolls.add(key)
                    if transform_node.inputs:
                        for input_pcoll in transform_node.inputs:
                            key = self._pin.cache_key(input_pcoll)
                            if key in self._pin._cached_pcoll_read:
                                self.unbounded_pcolls.add(key)
            v = CacheableUnboundedPCollectionVisitor(self)
            pipeline.visit(v)
            output_tags = v.unbounded_pcolls
            if output_tags:
                output_pcolls = pipeline | test_stream.TestStream(output_tags=output_tags, coder=self._cache_manager._default_pcoder)
                for (tag, pcoll) in output_pcolls.items():
                    self._cached_pcoll_read[tag] = pcoll

        class ReadCacheWireVisitor(PipelineVisitor):
            """Visitor wires cache read as inputs to replace corresponding original
      input PCollections in pipeline.
      """

            def __init__(self, pin):
                if False:
                    while True:
                        i = 10
                'Initializes with a PipelineInstrument.'
                self._pin = pin

            def enter_composite_transform(self, transform_node):
                if False:
                    while True:
                        i = 10
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    print('Hello World!')
                if transform_node.inputs:
                    main_inputs = dict(transform_node.main_inputs)
                    for (tag, input_pcoll) in main_inputs.items():
                        key = self._pin.cache_key(input_pcoll)
                        if key in self._pin._cached_pcoll_read:
                            self._pin._ignored_targets.add(input_pcoll)
                            main_inputs[tag] = self._pin._cached_pcoll_read[key]
                    transform_node.main_inputs = main_inputs
        v = ReadCacheWireVisitor(self)
        pipeline.visit(v)

    def _cacheable_inputs(self, transform):
        if False:
            while True:
                i = 10
        inputs = set()
        for in_pcoll in transform.inputs:
            if self.pcoll_id(in_pcoll) in self._cacheables:
                inputs.add(in_pcoll)
        return inputs

    def _all_inputs_outputs(self, transform):
        if False:
            i = 10
            return i + 15
        inputs = set()
        outputs = set()
        for in_pcoll in transform.inputs:
            inputs.add(in_pcoll)
        for (_, out_pcoll) in transform.outputs.items():
            outputs.add(out_pcoll)
        return (inputs, outputs)

    def pcoll_id(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        "Gets the PCollection id of the given pcoll.\n\n    Returns '' if not found.\n    "
        return self._pcoll_to_pcoll_id.get(str(pcoll), '')

    def cache_key(self, pcoll):
        if False:
            return 10
        "Gets the identifier of a cacheable PCollection in cache.\n\n    If the pcoll is not a cacheable, return ''.\n    This is only needed in pipeline instrument when the origin of given pcoll\n    is unknown (whether it's from the user pipeline or a runner pipeline). If\n    a pcoll is from the user pipeline, always use CacheKey.from_pcoll to build\n    the key.\n    The key is what the pcoll would use as identifier if it's materialized in\n    cache. It doesn't mean that there would definitely be such cache already.\n    Also, the pcoll can come from the original user defined pipeline object or\n    an equivalent pcoll from a transformed copy of the original pipeline.\n    "
        cacheable = self._cacheables.get(self.pcoll_id(pcoll), None)
        if cacheable:
            if cacheable.pcoll in self.runner_pcoll_to_user_pcoll:
                user_pcoll = self.runner_pcoll_to_user_pcoll[cacheable.pcoll]
            else:
                user_pcoll = cacheable.pcoll
            return CacheKey.from_pcoll(cacheable.var, user_pcoll).to_str()
        return ''

def build_pipeline_instrument(pipeline, options=None):
    if False:
        while True:
            i = 10
    "Creates PipelineInstrument for a pipeline and its options with cache.\n\n  Throughout the process, the returned PipelineInstrument snapshots the given\n  pipeline and then mutates the pipeline. It's invoked by interactive components\n  such as the InteractiveRunner and the given pipeline should be implicitly\n  created runner pipelines instead of pipeline instances defined by the user.\n\n  This is the shorthand for doing 3 steps: 1) compute once for metadata of the\n  given runner pipeline and everything watched from user pipelines; 2) associate\n  info between the runner pipeline and its corresponding user pipeline,\n  eliminate data from other user pipelines if there are any; 3) mutate the\n  runner pipeline to apply interactivity.\n  "
    pi = PipelineInstrument(pipeline, options)
    pi.preprocess()
    pi.instrument()
    return pi

def pcoll_to_pcoll_id(pipeline, original_context):
    if False:
        for i in range(10):
            print('nop')
    'Returns a dict mapping PCollections string to PCollection IDs.\n\n  Using a PipelineVisitor to iterate over every node in the pipeline,\n  records the mapping from PCollections to PCollections IDs. This mapping\n  will be used to query cached PCollections.\n\n  Returns:\n    (dict from str to str) a dict mapping str(pcoll) to pcoll_id.\n  '

    class PCollVisitor(PipelineVisitor):
        """"A visitor that records input and output values to be replaced.

    Input and output values that should be updated are recorded in maps
    input_replacements and output_replacements respectively.

    We cannot update input and output values while visiting since that
    results in validation errors.
    """

        def __init__(self):
            if False:
                print('Hello World!')
            self.pcoll_to_pcoll_id = {}

        def enter_composite_transform(self, transform_node):
            if False:
                return 10
            self.visit_transform(transform_node)

        def visit_transform(self, transform_node):
            if False:
                while True:
                    i = 10
            for pcoll in transform_node.outputs.values():
                self.pcoll_to_pcoll_id[str(pcoll)] = original_context.pcollections.get_id(pcoll)
    v = PCollVisitor()
    pipeline.visit(v)
    return v.pcoll_to_pcoll_id