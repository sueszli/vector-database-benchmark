import collections
import logging
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import TypeVar
from apache_beam import coders
from apache_beam.coders.coder_impl import create_InputStream
from apache_beam.coders.coder_impl import create_OutputStream
from apache_beam.portability import common_urns
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners import common
from apache_beam.runners import pipeline_context
from apache_beam.runners import runner
from apache_beam.runners.portability.fn_api_runner import translations
from apache_beam.runners.portability.fn_api_runner import worker_handlers
from apache_beam.runners.worker import bundle_processor
from apache_beam.transforms import core
from apache_beam.transforms import trigger
from apache_beam.utils import windowed_value
T = TypeVar('T')
_LOGGER = logging.getLogger(__name__)

class TrivialRunner(runner.PipelineRunner):
    """A bare-bones batch Python pipeline runner illistrating how to use the
  RunnerAPI and FnAPI to execute pipelines.

  Note that this runner is primarily for pedagogical purposes and is missing
  several features in order to keep it as simple as possible.  Where possible
  pointers are provided which this should serve as a useful starting point.
  """

    def run_portable_pipeline(self, pipeline, options):
        if False:
            while True:
                i = 10
        self.check_requirements(pipeline, self.supported_requirements())
        optimized_pipeline = translations.optimize_pipeline(pipeline, phases=translations.standard_optimize_phases(), known_runner_urns=frozenset([common_urns.primitives.IMPULSE.urn, common_urns.primitives.FLATTEN.urn, common_urns.primitives.GROUP_BY_KEY.urn]), partial=False)
        execution_state = ExecutionState(optimized_pipeline)
        for transform_id in optimized_pipeline.root_transform_ids:
            self.execute_transform(transform_id, execution_state)
        return runner.PipelineResult(runner.PipelineState.DONE)

    def execute_transform(self, transform_id, execution_state):
        if False:
            i = 10
            return i + 15
        'Execute a single transform.'
        transform_proto = execution_state.optimized_pipeline.components.transforms[transform_id]
        _LOGGER.info('Executing stage %s %s', transform_id, transform_proto.unique_name)
        if not is_primitive_transform(transform_proto):
            for sub_transform in transform_proto.subtransforms:
                self.execute_transform(sub_transform, execution_state)
        elif transform_proto.spec.urn == common_urns.primitives.IMPULSE.urn:
            execution_state.set_pcollection_contents(only_element(transform_proto.outputs.values()), [common.ENCODED_IMPULSE_VALUE])
        elif transform_proto.spec.urn == common_urns.primitives.FLATTEN.urn:
            output_pcoll_id = only_element(transform_proto.outputs.values())
            execution_state.set_pcollection_contents(output_pcoll_id, sum([execution_state.get_pcollection_contents(pc) for pc in transform_proto.inputs.values()], []))
        elif transform_proto.spec.urn == 'beam:runner:executable_stage:v1':
            self.execute_executable_stage(transform_proto, execution_state)
        elif transform_proto.spec.urn == common_urns.primitives.GROUP_BY_KEY.urn:
            self.group_by_key_and_window(only_element(transform_proto.inputs.values()), only_element(transform_proto.outputs.values()), execution_state)
        else:
            raise RuntimeError(f'Unsupported transform {transform_id} of type {{transform_proto.spec.urn}}')

    def execute_executable_stage(self, transform_proto, execution_state):
        if False:
            print('Hello World!')
        stage = beam_runner_api_pb2.ExecutableStagePayload.FromString(transform_proto.spec.payload)
        if stage.side_inputs:
            raise NotImplementedError()
        stage_transforms = {id: stage.components.transforms[id] for id in stage.transforms}
        input_transform = execution_state.new_id('stage_input')
        input_pcoll = stage.input
        stage_transforms[input_transform] = beam_runner_api_pb2.PTransform(spec=beam_runner_api_pb2.FunctionSpec(urn=bundle_processor.DATA_INPUT_URN, payload=beam_fn_api_pb2.RemoteGrpcPort(coder_id=execution_state.windowed_coder_id(stage.input)).SerializeToString()), outputs={'out': input_pcoll})
        output_ops_to_pcoll = {}
        for output_pcoll in stage.outputs:
            output_transform = execution_state.new_id('stage_output')
            stage_transforms[output_transform] = beam_runner_api_pb2.PTransform(spec=beam_runner_api_pb2.FunctionSpec(urn=bundle_processor.DATA_OUTPUT_URN, payload=beam_fn_api_pb2.RemoteGrpcPort(coder_id=execution_state.windowed_coder_id(output_pcoll)).SerializeToString()), inputs={'input': output_pcoll})
            output_ops_to_pcoll[output_transform] = output_pcoll
        process_bundle_descriptor = beam_fn_api_pb2.ProcessBundleDescriptor(id=execution_state.new_id('descriptor'), transforms=stage_transforms, pcollections=stage.components.pcollections, coders=execution_state.optimized_pipeline.components.coders, windowing_strategies=stage.components.windowing_strategies, environments=stage.components.environments)
        execution_state.register_process_bundle_descriptor(process_bundle_descriptor)
        process_bundle_id = execution_state.new_id('bundle')
        to_worker = execution_state.worker_handler.data_conn.output_stream(process_bundle_id, input_transform)
        for encoded_data in execution_state.get_pcollection_contents(input_pcoll):
            to_worker.write(encoded_data)
        to_worker.close()
        process_bundle_request = beam_fn_api_pb2.InstructionRequest(instruction_id=process_bundle_id, process_bundle=beam_fn_api_pb2.ProcessBundleRequest(process_bundle_descriptor_id=process_bundle_descriptor.id))
        result_future = execution_state.worker_handler.control_conn.push(process_bundle_request)
        for output in execution_state.worker_handler.data_conn.input_elements(process_bundle_id, list(output_ops_to_pcoll.keys())):
            if isinstance(output, beam_fn_api_pb2.Elements.Data):
                execution_state.set_pcollection_contents(output_ops_to_pcoll[output.transform_id], [output.data])
            else:
                raise RuntimeError('Unexpected data type: %s' % output)
        result = result_future.get()
        if result.error:
            raise RuntimeError(result.error)
        if result.process_bundle.residual_roots:
            raise NotImplementedError('SDF continuation')
        if result.process_bundle.requires_finalization:
            raise NotImplementedError('finalization')
        if result.process_bundle.elements.data:
            raise NotImplementedError('control-channel data')
        if result.process_bundle.elements.timers:
            raise NotImplementedError('timers')

    def group_by_key_and_window(self, input_pcoll, output_pcoll, execution_state):
        if False:
            i = 10
            return i + 15
        'Groups the elements of input_pcoll, placing their output in output_pcoll.\n    '
        input_coder = execution_state.windowed_coder(input_pcoll)
        key_coder = input_coder.key_coder()
        input_elements = []
        for encoded_elements in execution_state.get_pcollection_contents(input_pcoll):
            for element in decode_all(encoded_elements, input_coder):
                input_elements.append(element)
        components = execution_state.optimized_pipeline.components
        windowing = components.windowing_strategies[components.pcollections[input_pcoll].windowing_strategy_id]
        if windowing.merge_status == beam_runner_api_pb2.MergeStatus.Enum.NON_MERGING and windowing.output_time == beam_runner_api_pb2.OutputTime.Enum.END_OF_WINDOW:
            grouped = collections.defaultdict(list)
            for element in input_elements:
                for window in element.windows:
                    (key, value) = element.value
                    grouped[window, key_coder.encode(key)].append(value)
            output_elements = [windowed_value.WindowedValue((key_coder.decode(encoded_key), values), window.end, [window], trigger.BatchGlobalTriggerDriver.ONLY_FIRING) for ((window, encoded_key), values) in grouped.items()]
        else:
            trigger_driver = trigger.create_trigger_driver(execution_state.windowing_strategy(input_pcoll), True)
            grouped_by_key = collections.defaultdict(list)
            for element in input_elements:
                (key, value) = element.value
                grouped_by_key[key_coder.encode(key)].append(element.with_value(value))
            output_elements = []
            for (encoded_key, windowed_values) in grouped_by_key.items():
                for grouping in trigger_driver.process_entire_key(key_coder.decode(encoded_key), windowed_values):
                    output_elements.append(grouping)
        output_coder = execution_state.windowed_coder(output_pcoll)
        execution_state.set_pcollection_contents(output_pcoll, [encode_all(output_elements, output_coder)])

    def supported_requirements(self) -> Iterable[str]:
        if False:
            while True:
                i = 10
        return []

class ExecutionState:
    """A helper class holding various values and context during execution."""

    def __init__(self, optimized_pipeline):
        if False:
            while True:
                i = 10
        self.optimized_pipeline = optimized_pipeline
        self._pcollections_to_encoded_chunks = {}
        self._counter = 0
        self._process_bundle_descriptors = {}
        self.worker_handler = worker_handlers.EmbeddedWorkerHandler(None, state=worker_handlers.StateServicer(), provision_info=None, worker_manager=self)
        self._windowed_coders = {}
        for pcoll_id in self.optimized_pipeline.components.pcollections.keys():
            self.windowed_coder_id(pcoll_id)
        self._pipeline_context = pipeline_context.PipelineContext(optimized_pipeline.components)

    def register_process_bundle_descriptor(self, process_bundle_descriptor: beam_fn_api_pb2.ProcessBundleDescriptor):
        if False:
            return 10
        self._process_bundle_descriptors[process_bundle_descriptor.id] = process_bundle_descriptor

    def get_pcollection_contents(self, pcoll_id: str) -> List[bytes]:
        if False:
            print('Hello World!')
        return self._pcollections_to_encoded_chunks[pcoll_id]

    def set_pcollection_contents(self, pcoll_id: str, chunks: List[bytes]):
        if False:
            return 10
        self._pcollections_to_encoded_chunks[pcoll_id] = chunks

    def new_id(self, prefix='') -> str:
        if False:
            return 10
        self._counter += 1
        return f'runner_{prefix}_{self._counter}'

    def windowed_coder(self, pcollection_id: str) -> coders.Coder:
        if False:
            while True:
                i = 10
        return self._pipeline_context.coders.get_by_id(self.windowed_coder_id(pcollection_id))

    def windowing_strategy(self, pcollection_id: str) -> core.Windowing:
        if False:
            return 10
        return self._pipeline_context.windowing_strategies.get_by_id(self.optimized_pipeline.components.pcollections[pcollection_id].windowing_strategy_id)

    def windowed_coder_id(self, pcollection_id: str) -> str:
        if False:
            while True:
                i = 10
        pcoll = self.optimized_pipeline.components.pcollections[pcollection_id]
        windowing = self.optimized_pipeline.components.windowing_strategies[pcoll.windowing_strategy_id]
        return self._windowed_coder_id_from(pcoll.coder_id, windowing.window_coder_id)

    def _windowed_coder_id_from(self, coder_id: str, window_coder_id: str) -> str:
        if False:
            while True:
                i = 10
        if (coder_id, window_coder_id) not in self._windowed_coders:
            windowed_coder_id = self.new_id('windowed_coder')
            self._windowed_coders[coder_id, window_coder_id] = windowed_coder_id
            self.optimized_pipeline.components.coders[windowed_coder_id].CopyFrom(beam_runner_api_pb2.Coder(spec=beam_runner_api_pb2.FunctionSpec(urn=common_urns.coders.WINDOWED_VALUE.urn), component_coder_ids=[coder_id, window_coder_id]))
        return self._windowed_coders[coder_id, window_coder_id]

def is_primitive_transform(transform: beam_runner_api_pb2.PTransform) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return not transform.subtransforms and (not transform.outputs) or bool(set(transform.outputs.values()) - set(transform.inputs.values()))

def only_element(iterable: Iterable[T]) -> T:
    if False:
        for i in range(10):
            print('nop')
    (element,) = iterable
    return element

def decode_all(encoded_elements: bytes, coder: coders.Coder) -> Iterator[Any]:
    if False:
        return 10
    coder_impl = coder.get_impl()
    input_stream = create_InputStream(encoded_elements)
    while input_stream.size() > 0:
        yield coder_impl.decode_from_stream(input_stream, True)

def encode_all(elements: Iterator[T], coder: coders.Coder) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    coder_impl = coder.get_impl()
    output_stream = create_OutputStream()
    for element in elements:
        coder_impl.encode_to_stream(element, output_stream, True)
    return output_stream.get()