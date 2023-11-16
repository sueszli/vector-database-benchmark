"""DirectRunner, executing on the local machine.

The DirectRunner is a runner implementation that executes the entire
graph of transformations belonging to a pipeline on the local machine.
"""
import itertools
import logging
import time
import typing
from google.protobuf import wrappers_pb2
import apache_beam as beam
from apache_beam import coders
from apache_beam import typehints
from apache_beam.internal.util import ArgumentPlaceholder
from apache_beam.options.pipeline_options import DirectOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.value_provider import RuntimeValueProvider
from apache_beam.pvalue import PCollection
from apache_beam.runners.direct.bundle_factory import BundleFactory
from apache_beam.runners.direct.clock import RealClock
from apache_beam.runners.direct.clock import TestClock
from apache_beam.runners.runner import PipelineResult
from apache_beam.runners.runner import PipelineRunner
from apache_beam.runners.runner import PipelineState
from apache_beam.transforms import userstate
from apache_beam.transforms.core import CombinePerKey
from apache_beam.transforms.core import CombineValuesDoFn
from apache_beam.transforms.core import DoFn
from apache_beam.transforms.core import ParDo
from apache_beam.transforms.ptransform import PTransform
from apache_beam.transforms.timeutil import TimeDomain
from apache_beam.typehints import trivial_inference
__all__ = ['BundleBasedDirectRunner', 'DirectRunner', 'SwitchingDirectRunner']
_LOGGER = logging.getLogger(__name__)

class SwitchingDirectRunner(PipelineRunner):
    """Executes a single pipeline on the local machine.

  This implementation switches between using the FnApiRunner (which has
  high throughput for batch jobs) and using the BundleBasedDirectRunner,
  which supports streaming execution and certain primitives not yet
  implemented in the FnApiRunner.
  """

    def is_fnapi_compatible(self):
        if False:
            while True:
                i = 10
        return BundleBasedDirectRunner.is_fnapi_compatible()

    def run_pipeline(self, pipeline, options):
        if False:
            for i in range(10):
                print('nop')
        from apache_beam.pipeline import PipelineVisitor
        from apache_beam.testing.test_stream import TestStream
        from apache_beam.io.gcp.pubsub import ReadFromPubSub
        from apache_beam.io.gcp.pubsub import WriteToPubSub

        class _FnApiRunnerSupportVisitor(PipelineVisitor):
            """Visitor determining if a Pipeline can be run on the FnApiRunner."""

            def accept(self, pipeline):
                if False:
                    i = 10
                    return i + 15
                self.supported_by_fnapi_runner = True
                pipeline.visit(self)
                return self.supported_by_fnapi_runner

            def enter_composite_transform(self, applied_ptransform):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(applied_ptransform.transform, (ReadFromPubSub, WriteToPubSub)):
                    self.supported_by_fnapi_runner = False

            def visit_transform(self, applied_ptransform):
                if False:
                    return 10
                transform = applied_ptransform.transform
                if isinstance(transform, TestStream):
                    self.supported_by_fnapi_runner = False
                if isinstance(transform, beam.ParDo):
                    dofn = transform.dofn
                    if isinstance(dofn, CombineValuesDoFn):
                        (args, kwargs) = transform.raw_side_inputs
                        args_to_check = itertools.chain(args, kwargs.values())
                        if any((isinstance(arg, ArgumentPlaceholder) for arg in args_to_check)):
                            self.supported_by_fnapi_runner = False
                    if userstate.is_stateful_dofn(dofn):
                        (_, timer_specs) = userstate.get_dofn_specs(dofn)
                        for timer in timer_specs:
                            if timer.time_domain == TimeDomain.REAL_TIME:
                                self.supported_by_fnapi_runner = False
        if _FnApiRunnerSupportVisitor().accept(pipeline):
            from apache_beam.portability.api import beam_provision_api_pb2
            from apache_beam.runners.portability.fn_api_runner import fn_runner
            from apache_beam.runners.portability.portable_runner import JobServiceHandle
            all_options = options.get_all_options()
            encoded_options = JobServiceHandle.encode_pipeline_options(all_options)
            provision_info = fn_runner.ExtendedProvisionInfo(beam_provision_api_pb2.ProvisionInfo(pipeline_options=encoded_options))
            runner = fn_runner.FnApiRunner(provision_info=provision_info)
        else:
            runner = BundleBasedDirectRunner()
        return runner.run_pipeline(pipeline, options)
K = typing.TypeVar('K')
V = typing.TypeVar('V')

@typehints.with_input_types(typing.Tuple[K, V])
@typehints.with_output_types(typing.Tuple[K, typing.Iterable[V]])
class _GroupByKeyOnly(PTransform):
    """A group by key transform, ignoring windows."""

    def infer_output_type(self, input_type):
        if False:
            i = 10
            return i + 15
        (key_type, value_type) = trivial_inference.key_value_types(input_type)
        return typehints.KV[key_type, typehints.Iterable[value_type]]

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        self._check_pcollection(pcoll)
        return PCollection.from_(pcoll)

@typehints.with_input_types(typing.Tuple[K, typing.Iterable[V]])
@typehints.with_output_types(typing.Tuple[K, typing.Iterable[V]])
class _GroupAlsoByWindow(ParDo):
    """The GroupAlsoByWindow transform."""

    def __init__(self, windowing):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(_GroupAlsoByWindowDoFn(windowing))
        self.windowing = windowing

    def expand(self, pcoll):
        if False:
            return 10
        self._check_pcollection(pcoll)
        return PCollection.from_(pcoll)

class _GroupAlsoByWindowDoFn(DoFn):

    def __init__(self, windowing):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.windowing = windowing

    def infer_output_type(self, input_type):
        if False:
            while True:
                i = 10
        (key_type, windowed_value_iter_type) = trivial_inference.key_value_types(input_type)
        value_type = windowed_value_iter_type.inner_type.inner_type
        return typehints.KV[key_type, typehints.Iterable[value_type]]

    def start_bundle(self):
        if False:
            i = 10
            return i + 15
        from apache_beam.transforms.trigger import create_trigger_driver
        self.driver = create_trigger_driver(self.windowing, True)

    def process(self, element):
        if False:
            print('Hello World!')
        (k, vs) = element
        return self.driver.process_entire_key(k, vs)

@typehints.with_input_types(typing.Tuple[K, V])
@typehints.with_output_types(typing.Tuple[K, typing.Iterable[V]])
class _StreamingGroupByKeyOnly(_GroupByKeyOnly):
    """Streaming GroupByKeyOnly placeholder for overriding in DirectRunner."""
    urn = 'direct_runner:streaming_gbko:v0.1'

    def to_runner_api_parameter(self, unused_context):
        if False:
            while True:
                i = 10
        return (_StreamingGroupByKeyOnly.urn, None)

    @staticmethod
    @PTransform.register_urn(urn, None)
    def from_runner_api_parameter(unused_ptransform, unused_payload, unused_context):
        if False:
            i = 10
            return i + 15
        return _StreamingGroupByKeyOnly()

@typehints.with_input_types(typing.Tuple[K, typing.Iterable[V]])
@typehints.with_output_types(typing.Tuple[K, typing.Iterable[V]])
class _StreamingGroupAlsoByWindow(_GroupAlsoByWindow):
    """Streaming GroupAlsoByWindow placeholder for overriding in DirectRunner."""
    urn = 'direct_runner:streaming_gabw:v0.1'

    def to_runner_api_parameter(self, context):
        if False:
            while True:
                i = 10
        return (_StreamingGroupAlsoByWindow.urn, wrappers_pb2.BytesValue(value=context.windowing_strategies.get_id(self.windowing)))

    @staticmethod
    @PTransform.register_urn(urn, wrappers_pb2.BytesValue)
    def from_runner_api_parameter(unused_ptransform, payload, context):
        if False:
            return 10
        return _StreamingGroupAlsoByWindow(context.windowing_strategies.get_by_id(payload.value))

@typehints.with_input_types(typing.Tuple[K, typing.Iterable[V]])
@typehints.with_output_types(typing.Tuple[K, typing.Iterable[V]])
class _GroupByKey(PTransform):
    """The DirectRunner GroupByKey implementation."""

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        from apache_beam.coders import typecoders
        input_type = pcoll.element_type
        if input_type is not None:
            (key_type, value_type) = trivial_inference.key_value_types(input_type)
            pcoll.element_type = typehints.typehints.coerce_to_kv_type(pcoll.element_type)
            typecoders.registry.verify_deterministic(typecoders.registry.get_coder(key_type), 'GroupByKey operation "%s"' % self.label)
            reify_output_type = typehints.KV[key_type, typehints.WindowedValue[value_type]]
            gbk_input_type = typehints.KV[key_type, typehints.Iterable[typehints.WindowedValue[value_type]]]
            gbk_output_type = typehints.KV[key_type, typehints.Iterable[value_type]]
            return pcoll | 'ReifyWindows' >> ParDo(beam.GroupByKey.ReifyWindows()).with_output_types(reify_output_type) | 'GroupByKey' >> _GroupByKeyOnly().with_input_types(reify_output_type).with_output_types(gbk_input_type) | 'GroupByWindow' >> _GroupAlsoByWindow(pcoll.windowing).with_input_types(gbk_input_type).with_output_types(gbk_output_type)
        else:
            return pcoll | 'ReifyWindows' >> ParDo(beam.GroupByKey.ReifyWindows()) | 'GroupByKey' >> _GroupByKeyOnly() | 'GroupByWindow' >> _GroupAlsoByWindow(pcoll.windowing)

def _get_transform_overrides(pipeline_options):
    if False:
        for i in range(10):
            print('nop')
    from apache_beam.pipeline import PTransformOverride
    from apache_beam.runners.direct.helper_transforms import LiftedCombinePerKey
    from apache_beam.runners.direct.sdf_direct_runner import ProcessKeyedElementsViaKeyedWorkItemsOverride
    from apache_beam.runners.direct.sdf_direct_runner import SplittableParDoOverride

    class CombinePerKeyOverride(PTransformOverride):

        def matches(self, applied_ptransform):
            if False:
                print('Hello World!')
            if isinstance(applied_ptransform.transform, CombinePerKey):
                return applied_ptransform.inputs[0].windowing.is_default()

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                for i in range(10):
                    print('nop')
            try:
                transform = applied_ptransform.transform
                return LiftedCombinePerKey(transform.fn, transform.args, transform.kwargs)
            except NotImplementedError:
                return transform

    class StreamingGroupByKeyOverride(PTransformOverride):

        def matches(self, applied_ptransform):
            if False:
                while True:
                    i = 10
            return applied_ptransform.transform.__class__ == _GroupByKeyOnly

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                i = 10
                return i + 15
            transform = _StreamingGroupByKeyOnly()
            return transform

    class StreamingGroupAlsoByWindowOverride(PTransformOverride):

        def matches(self, applied_ptransform):
            if False:
                return 10
            transform = applied_ptransform.transform
            return isinstance(applied_ptransform.transform, ParDo) and isinstance(transform.dofn, _GroupAlsoByWindowDoFn) and (transform.__class__ != _StreamingGroupAlsoByWindow)

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                for i in range(10):
                    print('nop')
            transform = _StreamingGroupAlsoByWindow(applied_ptransform.transform.dofn.windowing)
            return transform

    class TestStreamOverride(PTransformOverride):

        def matches(self, applied_ptransform):
            if False:
                i = 10
                return i + 15
            from apache_beam.testing.test_stream import TestStream
            self.applied_ptransform = applied_ptransform
            return isinstance(applied_ptransform.transform, TestStream)

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                return 10
            from apache_beam.runners.direct.test_stream_impl import _ExpandableTestStream
            return _ExpandableTestStream(applied_ptransform.transform)

    class GroupByKeyPTransformOverride(PTransformOverride):
        """A ``PTransformOverride`` for ``GroupByKey``.

    This replaces the Beam implementation as a primitive.
    """

        def matches(self, applied_ptransform):
            if False:
                return 10
            from apache_beam.transforms.core import GroupByKey
            return isinstance(applied_ptransform.transform, GroupByKey)

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                while True:
                    i = 10
            return _GroupByKey()
    overrides = [GroupByKeyPTransformOverride(), SplittableParDoOverride(), ProcessKeyedElementsViaKeyedWorkItemsOverride(), CombinePerKeyOverride(), TestStreamOverride()]
    if pipeline_options.view_as(StandardOptions).streaming:
        overrides.append(StreamingGroupByKeyOverride())
        overrides.append(StreamingGroupAlsoByWindowOverride())
    try:
        from apache_beam.io.gcp import pubsub as unused_pubsub
        overrides += _get_pubsub_transform_overrides(pipeline_options)
    except ImportError:
        pass
    overrides.append(GroupByKeyPTransformOverride())
    return overrides

class _DirectReadFromPubSub(PTransform):

    def __init__(self, source):
        if False:
            i = 10
            return i + 15
        self._source = source

    def _infer_output_coder(self, unused_input_type=None, unused_input_coder=None):
        if False:
            print('Hello World!')
        return coders.BytesCoder()

    def get_windowing(self, unused_inputs):
        if False:
            print('Hello World!')
        return beam.Windowing(beam.window.GlobalWindows())

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        return PCollection(self.pipeline, is_bounded=self._source.is_bounded())

class _DirectWriteToPubSubFn(DoFn):
    BUFFER_SIZE_ELEMENTS = 100
    FLUSH_TIMEOUT_SECS = BUFFER_SIZE_ELEMENTS * 0.5

    def __init__(self, transform):
        if False:
            print('Hello World!')
        self.project = transform.project
        self.short_topic_name = transform.topic_name
        self.id_label = transform.id_label
        self.timestamp_attribute = transform.timestamp_attribute
        self.with_attributes = transform.with_attributes
        if transform.id_label:
            raise NotImplementedError('DirectRunner: id_label is not supported for PubSub writes')
        if transform.timestamp_attribute:
            raise NotImplementedError('DirectRunner: timestamp_attribute is not supported for PubSub writes')

    def start_bundle(self):
        if False:
            print('Hello World!')
        self._buffer = []

    def process(self, elem):
        if False:
            i = 10
            return i + 15
        self._buffer.append(elem)
        if len(self._buffer) >= self.BUFFER_SIZE_ELEMENTS:
            self._flush()

    def finish_bundle(self):
        if False:
            return 10
        self._flush()

    def _flush(self):
        if False:
            i = 10
            return i + 15
        from google.cloud import pubsub
        pub_client = pubsub.PublisherClient()
        topic = pub_client.topic_path(self.project, self.short_topic_name)
        if self.with_attributes:
            futures = [pub_client.publish(topic, elem.data, **elem.attributes) for elem in self._buffer]
        else:
            futures = [pub_client.publish(topic, elem) for elem in self._buffer]
        timer_start = time.time()
        for future in futures:
            remaining = self.FLUSH_TIMEOUT_SECS - (time.time() - timer_start)
            future.result(remaining)
        self._buffer = []

def _get_pubsub_transform_overrides(pipeline_options):
    if False:
        for i in range(10):
            print('nop')
    from apache_beam.io.gcp import pubsub as beam_pubsub
    from apache_beam.pipeline import PTransformOverride

    class ReadFromPubSubOverride(PTransformOverride):

        def matches(self, applied_ptransform):
            if False:
                while True:
                    i = 10
            return isinstance(applied_ptransform.transform, beam_pubsub.ReadFromPubSub)

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                i = 10
                return i + 15
            if not pipeline_options.view_as(StandardOptions).streaming:
                raise Exception('PubSub I/O is only available in streaming mode (use the --streaming flag).')
            return _DirectReadFromPubSub(applied_ptransform.transform._source)

    class WriteToPubSubOverride(PTransformOverride):

        def matches(self, applied_ptransform):
            if False:
                print('Hello World!')
            return isinstance(applied_ptransform.transform, beam_pubsub.WriteToPubSub)

        def get_replacement_transform_for_applied_ptransform(self, applied_ptransform):
            if False:
                while True:
                    i = 10
            if not pipeline_options.view_as(StandardOptions).streaming:
                raise Exception('PubSub I/O is only available in streaming mode (use the --streaming flag).')
            return beam.ParDo(_DirectWriteToPubSubFn(applied_ptransform.transform))
    return [ReadFromPubSubOverride(), WriteToPubSubOverride()]

class BundleBasedDirectRunner(PipelineRunner):
    """Executes a single pipeline on the local machine."""

    @staticmethod
    def is_fnapi_compatible():
        if False:
            print('Hello World!')
        return False

    def run_pipeline(self, pipeline, options):
        if False:
            return 10
        'Execute the entire pipeline and returns an DirectPipelineResult.'
        from apache_beam.pipeline import PipelineVisitor
        from apache_beam.runners.direct.consumer_tracking_pipeline_visitor import ConsumerTrackingPipelineVisitor
        from apache_beam.runners.direct.evaluation_context import EvaluationContext
        from apache_beam.runners.direct.executor import Executor
        from apache_beam.runners.direct.transform_evaluator import TransformEvaluatorRegistry
        from apache_beam.testing.test_stream import TestStream

        class TestStreamUsageVisitor(PipelineVisitor):
            """Visitor determining whether a Pipeline uses a TestStream."""

            def __init__(self):
                if False:
                    return 10
                self.uses_test_stream = False

            def visit_transform(self, applied_ptransform):
                if False:
                    i = 10
                    return i + 15
                if isinstance(applied_ptransform.transform, TestStream):
                    self.uses_test_stream = True
        visitor = TestStreamUsageVisitor()
        pipeline.visit(visitor)
        clock = TestClock() if visitor.uses_test_stream else RealClock()
        pipeline.replace_all(_get_transform_overrides(options))
        _LOGGER.info('Running pipeline with DirectRunner.')
        self.consumer_tracking_visitor = ConsumerTrackingPipelineVisitor()
        pipeline.visit(self.consumer_tracking_visitor)
        evaluation_context = EvaluationContext(options, BundleFactory(stacked=options.view_as(DirectOptions).direct_runner_use_stacked_bundle), self.consumer_tracking_visitor.root_transforms, self.consumer_tracking_visitor.value_to_consumers, self.consumer_tracking_visitor.step_names, self.consumer_tracking_visitor.views, clock)
        executor = Executor(self.consumer_tracking_visitor.value_to_consumers, TransformEvaluatorRegistry(evaluation_context), evaluation_context)
        RuntimeValueProvider.set_runtime_options({})
        executor.start(self.consumer_tracking_visitor.root_transforms)
        result = DirectPipelineResult(executor, evaluation_context)
        return result
DirectRunner = SwitchingDirectRunner

class DirectPipelineResult(PipelineResult):
    """A DirectPipelineResult provides access to info about a pipeline."""

    def __init__(self, executor, evaluation_context):
        if False:
            print('Hello World!')
        super().__init__(PipelineState.RUNNING)
        self._executor = executor
        self._evaluation_context = evaluation_context

    def __del__(self):
        if False:
            print('Hello World!')
        if self._state == PipelineState.RUNNING:
            _LOGGER.warning('The DirectPipelineResult is being garbage-collected while the DirectRunner is still running the corresponding pipeline. This may lead to incomplete execution of the pipeline if the main thread exits before pipeline completion. Consider using result.wait_until_finish() to wait for completion of pipeline execution.')

    def wait_until_finish(self, duration=None):
        if False:
            while True:
                i = 10
        if not PipelineState.is_terminal(self.state):
            if duration:
                raise NotImplementedError('DirectRunner does not support duration argument.')
            try:
                self._executor.await_completion()
                self._state = PipelineState.DONE
            except:
                self._state = PipelineState.FAILED
                raise
        return self._state

    def aggregated_values(self, aggregator_or_name):
        if False:
            i = 10
            return i + 15
        return self._evaluation_context.get_aggregator_values(aggregator_or_name)

    def metrics(self):
        if False:
            print('Hello World!')
        return self._evaluation_context.metrics()

    def cancel(self):
        if False:
            i = 10
            return i + 15
        'Shuts down pipeline workers.\n\n    For testing use only. Does not properly wait for pipeline workers to shut\n    down.\n    '
        self._state = PipelineState.CANCELLING
        self._executor.shutdown()
        self._state = PipelineState.CANCELLED