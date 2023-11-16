import abc
from abc import abstractmethod
from typing import Iterable, Any, Dict, List
from apache_beam.runners.worker.bundle_processor import TimerInfo, DataOutputOperation
from apache_beam.runners.worker.operations import Operation
from apache_beam.utils import windowed_value
from apache_beam.utils.windowed_value import WindowedValue
from pyflink.common.constants import DEFAULT_OUTPUT_TAG
from pyflink.fn_execution.flink_fn_execution_pb2 import UserDefinedDataStreamFunction
from pyflink.fn_execution.table.operations import BundleOperation
from pyflink.fn_execution.profiler import Profiler

class OutputProcessor(abc.ABC):

    @abstractmethod
    def process_outputs(self, windowed_value: WindowedValue, results: Iterable[Any]):
        if False:
            i = 10
            return i + 15
        pass

    def close(self):
        if False:
            while True:
                i = 10
        pass

class NetworkOutputProcessor(OutputProcessor):

    def __init__(self, consumer):
        if False:
            print('Hello World!')
        assert isinstance(consumer, DataOutputOperation)
        self._consumer = consumer
        self._value_coder_impl = consumer.windowed_coder.wrapped_value_coder.get_impl()._value_coder

    def process_outputs(self, windowed_value: WindowedValue, results: Iterable[Any]):
        if False:
            for i in range(10):
                print('nop')
        output_stream = self._consumer.output_stream
        self._value_coder_impl.encode_to_stream(results, output_stream, True)
        self._value_coder_impl._output_stream.maybe_flush()

    def close(self):
        if False:
            return 10
        self._value_coder_impl._output_stream.close()

class IntermediateOutputProcessor(OutputProcessor):

    def __init__(self, consumer):
        if False:
            return 10
        self._consumer = consumer

    def process_outputs(self, windowed_value: WindowedValue, results: Iterable[Any]):
        if False:
            i = 10
            return i + 15
        self._consumer.process(windowed_value.with_value(results))

class FunctionOperation(Operation):
    """
    Base class of function operation that will execute StatelessFunction or StatefulFunction for
    each input element.
    """

    def __init__(self, name, spec, counter_factory, sampler, consumers, operation_cls, operator_state_backend):
        if False:
            while True:
                i = 10
        super(FunctionOperation, self).__init__(name, spec, counter_factory, sampler)
        self._output_processors = self._create_output_processors(consumers)
        self.operation_cls = operation_cls
        self.operator_state_backend = operator_state_backend
        self.operation = self.generate_operation()
        self.process_element = self.operation.process_element
        self.operation.open()
        if spec.serialized_fn.profile_enabled:
            self._profiler = Profiler()
        else:
            self._profiler = None
        if isinstance(spec.serialized_fn, UserDefinedDataStreamFunction):
            self._has_side_output = spec.serialized_fn.has_side_output
        else:
            self._has_side_output = False
        if not self._has_side_output:
            self._main_output_processor = self._output_processors[DEFAULT_OUTPUT_TAG][0]

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        super(FunctionOperation, self).setup()

    def start(self):
        if False:
            print('Hello World!')
        with self.scoped_start_state:
            super(FunctionOperation, self).start()
            if self._profiler:
                self._profiler.start()

    def finish(self):
        if False:
            for i in range(10):
                print('nop')
        with self.scoped_finish_state:
            super(FunctionOperation, self).finish()
            self.operation.finish()
            if self._profiler:
                self._profiler.close()

    def needs_finalization(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        super(FunctionOperation, self).reset()

    def teardown(self):
        if False:
            return 10
        with self.scoped_finish_state:
            self.operation.close()
            for processors in self._output_processors.values():
                for p in processors:
                    p.close()

    def progress_metrics(self):
        if False:
            print('Hello World!')
        metrics = super(FunctionOperation, self).progress_metrics()
        metrics.processed_elements.measured.output_element_counts.clear()
        tag = None
        receiver = self.receivers[0]
        metrics.processed_elements.measured.output_element_counts[str(tag)] = receiver.opcounter.element_counter.value()
        return metrics

    def process(self, o: WindowedValue):
        if False:
            while True:
                i = 10
        with self.scoped_process_state:
            if self._has_side_output:
                for value in o.value:
                    for (tag, row) in self.process_element(value):
                        for p in self._output_processors.get(tag, []):
                            p.process_outputs(o, [row])
            elif isinstance(self.operation, BundleOperation):
                for value in o.value:
                    self.process_element(value)
                self._main_output_processor.process_outputs(o, self.operation.finish_bundle())
            else:
                for value in o.value:
                    self._main_output_processor.process_outputs(o, self.operation.process_element(value))

    def monitoring_infos(self, transform_id, tag_to_pcollection_id):
        if False:
            i = 10
            return i + 15
        '\n        Only pass user metric to Java\n        :param tag_to_pcollection_id: useless for user metric\n        '
        return super().user_monitoring_infos(transform_id)

    @staticmethod
    def _create_output_processors(consumers_map):
        if False:
            return 10

        def _create_processor(consumer):
            if False:
                i = 10
                return i + 15
            if isinstance(consumer, DataOutputOperation):
                return NetworkOutputProcessor(consumer)
            else:
                return IntermediateOutputProcessor(consumer)
        return {tag: [_create_processor(c) for c in consumers] for (tag, consumers) in consumers_map.items()}

    @abstractmethod
    def generate_operation(self):
        if False:
            while True:
                i = 10
        pass

class StatelessFunctionOperation(FunctionOperation):

    def __init__(self, name, spec, counter_factory, sampler, consumers, operation_cls, operator_state_backend):
        if False:
            print('Hello World!')
        super(StatelessFunctionOperation, self).__init__(name, spec, counter_factory, sampler, consumers, operation_cls, operator_state_backend)

    def generate_operation(self):
        if False:
            print('Hello World!')
        if self.operator_state_backend is not None:
            return self.operation_cls(self.spec.serialized_fn, self.operator_state_backend)
        else:
            return self.operation_cls(self.spec.serialized_fn)

class StatefulFunctionOperation(FunctionOperation):

    def __init__(self, name, spec, counter_factory, sampler, consumers, operation_cls, keyed_state_backend, operator_state_backend):
        if False:
            i = 10
            return i + 15
        self._keyed_state_backend = keyed_state_backend
        self._reusable_windowed_value = windowed_value.create(None, -1, None, None)
        super(StatefulFunctionOperation, self).__init__(name, spec, counter_factory, sampler, consumers, operation_cls, operator_state_backend)

    def generate_operation(self):
        if False:
            i = 10
            return i + 15
        if self.operator_state_backend is not None:
            return self.operation_cls(self.spec.serialized_fn, self._keyed_state_backend, self.operator_state_backend)
        else:
            return self.operation_cls(self.spec.serialized_fn, self._keyed_state_backend)

    def add_timer_info(self, timer_family_id: str, timer_info: TimerInfo):
        if False:
            print('Hello World!')
        self.operation.add_timer_info(timer_info)

    def process_timer(self, tag, timer_data):
        if False:
            print('Hello World!')
        if self._has_side_output:
            for (tag, row) in self.operation.process_timer(timer_data.user_key):
                for p in self._output_processors.get(tag, []):
                    p.process_outputs(self._reusable_windowed_value, [row])
        else:
            self._main_output_processor.process_outputs(self._reusable_windowed_value, self.operation.process_timer(timer_data.user_key))