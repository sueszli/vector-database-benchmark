import logging
import re
import unittest
import grpc
import apache_beam as beam
from apache_beam.coders.coders import FastPrimitivesCoder
from apache_beam.portability import common_urns
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.portability.api import endpoints_pb2
from apache_beam.runners import common
from apache_beam.runners.common import NameContext
from apache_beam.runners.worker import bundle_processor
from apache_beam.runners.worker import log_handler
from apache_beam.runners.worker import operations
from apache_beam.runners.worker import statesampler
from apache_beam.runners.worker.bundle_processor import BeamTransformFactory
from apache_beam.runners.worker.bundle_processor import BundleProcessor
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils import thread_pool_executor
from apache_beam.utils.windowed_value import WindowedValue
_LOGGER = logging.getLogger(__name__)

@BeamTransformFactory.register_urn('beam:internal:testexn:v1', bytes)
def create_exception_dofn(factory, transform_id, transform_proto, payload, consumers):
    if False:
        print('Hello World!')
    'Returns a test DoFn that raises the given exception.'

    class RaiseException(beam.DoFn):

        def __init__(self, msg):
            if False:
                print('Hello World!')
            self.msg = msg.decode()

        def process(self, _):
            if False:
                print('Hello World!')
            raise RuntimeError(self.msg)
    return bundle_processor._create_simple_pardo_operation(factory, transform_id, transform_proto, consumers, RaiseException(payload))

class TestOperation(operations.Operation):
    """Test operation that forwards its payload to consumers."""

    class Spec:

        def __init__(self, transform_proto):
            if False:
                for i in range(10):
                    print('nop')
            self.output_coders = [FastPrimitivesCoder() for _ in transform_proto.outputs]

    def __init__(self, transform_proto, name_context, counter_factory, state_sampler, consumers, payload):
        if False:
            print('Hello World!')
        super().__init__(name_context, self.Spec(transform_proto), counter_factory, state_sampler)
        self.payload = payload
        for (_, consumer_ops) in consumers.items():
            for consumer in consumer_ops:
                self.add_receiver(consumer, 0)

    def start(self):
        if False:
            i = 10
            return i + 15
        super().start()
        if self.payload:
            self.process(WindowedValue(self.payload, timestamp=0, windows=[GlobalWindow()]))

    def process(self, windowed_value):
        if False:
            while True:
                i = 10
        self.output(windowed_value)

@BeamTransformFactory.register_urn('beam:internal:testop:v1', bytes)
def create_test_op(factory, transform_id, transform_proto, payload, consumers):
    if False:
        while True:
            i = 10
    return TestOperation(transform_proto, common.NameContext(transform_proto.unique_name, transform_id), factory.counter_factory, factory.state_sampler, consumers, payload)

class BeamFnLoggingServicer(beam_fn_api_pb2_grpc.BeamFnLoggingServicer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.log_records_received = []

    def Logging(self, request_iterator, context):
        if False:
            while True:
                i = 10
        for log_record in request_iterator:
            self.log_records_received.append(log_record)
        yield beam_fn_api_pb2.LogControl()

class FnApiLogRecordHandlerTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_logging_service = BeamFnLoggingServicer()
        self.server = grpc.server(thread_pool_executor.shared_unbounded_instance())
        beam_fn_api_pb2_grpc.add_BeamFnLoggingServicer_to_server(self.test_logging_service, self.server)
        self.test_port = self.server.add_insecure_port('[::]:0')
        self.server.start()
        self.logging_service_descriptor = endpoints_pb2.ApiServiceDescriptor()
        self.logging_service_descriptor.url = 'localhost:%s' % self.test_port
        self.fn_log_handler = log_handler.FnApiLogRecordHandler(self.logging_service_descriptor)
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().addHandler(self.fn_log_handler)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.server.stop(5)

    def _verify_fn_log_handler(self, num_log_entries):
        if False:
            while True:
                i = 10
        msg = 'Testing fn logging'
        _LOGGER.debug('Debug Message 1')
        for idx in range(num_log_entries):
            _LOGGER.info('%s: %s', msg, idx)
        _LOGGER.debug('Debug Message 2')
        self.fn_log_handler.close()
        num_received_log_entries = 0
        for outer in self.test_logging_service.log_records_received:
            for log_entry in outer.log_entries:
                self.assertEqual(beam_fn_api_pb2.LogEntry.Severity.INFO, log_entry.severity)
                self.assertEqual('%s: %s' % (msg, num_received_log_entries), log_entry.message)
                self.assertTrue(re.match('.*log_handler_test.py:\\d+', log_entry.log_location), log_entry.log_location)
                self.assertGreater(log_entry.timestamp.seconds, 0)
                self.assertGreaterEqual(log_entry.timestamp.nanos, 0)
                num_received_log_entries += 1
        self.assertEqual(num_received_log_entries, num_log_entries)

    def assertContains(self, haystack, needle):
        if False:
            i = 10
            return i + 15
        self.assertTrue(needle in haystack, 'Expected %r to contain %r.' % (haystack, needle))

    def test_exc_info(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            raise ValueError('some message')
        except ValueError:
            _LOGGER.error('some error', exc_info=True)
        self.fn_log_handler.close()
        log_entry = self.test_logging_service.log_records_received[0].log_entries[0]
        self.assertContains(log_entry.message, 'some error')
        self.assertContains(log_entry.trace, 'some message')
        self.assertContains(log_entry.trace, 'log_handler_test.py')

    def test_format_bad_message(self):
        if False:
            while True:
                i = 10
        self.fn_log_handler.emit(logging.LogRecord('name', logging.ERROR, 'pathname', 777, 'TestLog %d', (None,), exc_info=None))
        self.fn_log_handler.close()
        log_entry = self.test_logging_service.log_records_received[0].log_entries[0]
        self.assertContains(log_entry.message, "Failed to format 'TestLog %d' with args '(None,)' during logging.")

    def test_context(self):
        if False:
            return 10
        try:
            with statesampler.instruction_id('A'):
                tracker = statesampler.for_test()
                with tracker.scoped_state(NameContext('name', 'tid'), 'stage'):
                    _LOGGER.info('message a')
            with statesampler.instruction_id('B'):
                _LOGGER.info('message b')
            _LOGGER.info('message c')
            self.fn_log_handler.close()
            (a, b, c) = sum([list(logs.log_entries) for logs in self.test_logging_service.log_records_received], [])
            self.assertEqual(a.instruction_id, 'A')
            self.assertEqual(b.instruction_id, 'B')
            self.assertEqual(c.instruction_id, '')
            self.assertEqual(a.transform_id, 'tid')
            self.assertEqual(b.transform_id, '')
            self.assertEqual(c.transform_id, '')
        finally:
            statesampler.set_current_tracker(None)

    def test_extracts_transform_id_during_exceptions(self):
        if False:
            print('Hello World!')
        'Tests that transform ids are captured during user code exceptions.'
        descriptor = beam_fn_api_pb2.ProcessBundleDescriptor()
        WINDOWING_ID = 'window'
        WINDOW_CODER_ID = 'cw'
        window = descriptor.windowing_strategies[WINDOWING_ID]
        window.window_fn.urn = common_urns.global_windows.urn
        window.window_coder_id = WINDOW_CODER_ID
        window.trigger.default.SetInParent()
        window_coder = descriptor.coders[WINDOW_CODER_ID]
        window_coder.spec.urn = common_urns.StandardCoders.Enum.GLOBAL_WINDOW.urn
        INPUT_PCOLLECTION_ID = 'pc-in'
        INPUT_CODER_ID = 'c-in'
        descriptor.pcollections[INPUT_PCOLLECTION_ID].unique_name = INPUT_PCOLLECTION_ID
        descriptor.pcollections[INPUT_PCOLLECTION_ID].coder_id = INPUT_CODER_ID
        descriptor.pcollections[INPUT_PCOLLECTION_ID].windowing_strategy_id = WINDOWING_ID
        descriptor.coders[INPUT_CODER_ID].spec.urn = common_urns.StandardCoders.Enum.BYTES.urn
        OUTPUT_PCOLLECTION_ID = 'pc-out'
        OUTPUT_CODER_ID = 'c-out'
        descriptor.pcollections[OUTPUT_PCOLLECTION_ID].unique_name = OUTPUT_PCOLLECTION_ID
        descriptor.pcollections[OUTPUT_PCOLLECTION_ID].coder_id = OUTPUT_CODER_ID
        descriptor.pcollections[OUTPUT_PCOLLECTION_ID].windowing_strategy_id = WINDOWING_ID
        descriptor.coders[OUTPUT_CODER_ID].spec.urn = common_urns.StandardCoders.Enum.BYTES.urn
        TEST_OP_TRANSFORM_ID = 'test_op'
        test_transform = descriptor.transforms[TEST_OP_TRANSFORM_ID]
        test_transform.outputs['None'] = INPUT_PCOLLECTION_ID
        test_transform.spec.urn = 'beam:internal:testop:v1'
        test_transform.spec.payload = b'hello, world!'
        TEST_EXCEPTION_TRANSFORM_ID = 'test_transform'
        test_transform = descriptor.transforms[TEST_EXCEPTION_TRANSFORM_ID]
        test_transform.inputs['0'] = INPUT_PCOLLECTION_ID
        test_transform.outputs['None'] = OUTPUT_PCOLLECTION_ID
        test_transform.spec.urn = 'beam:internal:testexn:v1'
        test_transform.spec.payload = b'expected exception'
        processor = BundleProcessor(descriptor, None, None)
        with self.assertRaisesRegex(RuntimeError, 'expected exception'):
            processor.process_bundle('instruction_id')
        self.fn_log_handler.close()
        logs = [log for logs in self.test_logging_service.log_records_received for log in logs.log_entries]
        actual_log = logs[0]
        self.assertEqual(actual_log.severity, beam_fn_api_pb2.LogEntry.Severity.ERROR)
        self.assertTrue('expected exception' in actual_log.message)
        self.assertEqual(actual_log.transform_id, 'test_transform')
data = {'one_batch': log_handler.FnApiLogRecordHandler._MAX_BATCH_SIZE - 47, 'exact_multiple': log_handler.FnApiLogRecordHandler._MAX_BATCH_SIZE, 'multi_batch': log_handler.FnApiLogRecordHandler._MAX_BATCH_SIZE * 3 + 47}

def _create_test(name, num_logs):
    if False:
        print('Hello World!')
    setattr(FnApiLogRecordHandlerTest, 'test_%s' % name, lambda self: self._verify_fn_log_handler(num_logs))
for (test_name, num_logs_entries) in data.items():
    _create_test(test_name, num_logs_entries)
if __name__ == '__main__':
    unittest.main()