"""Tests for apache_beam.runners.worker.data_plane."""
import itertools
import logging
import time
import unittest
import grpc
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.runners.worker import data_plane
from apache_beam.runners.worker.worker_id_interceptor import WorkerIdInterceptor
from apache_beam.utils import thread_pool_executor

class DataChannelTest(unittest.TestCase):

    def test_grpc_data_channel(self):
        if False:
            while True:
                i = 10
        self._grpc_data_channel_test()

    def test_time_based_flush_grpc_data_channel(self):
        if False:
            i = 10
            return i + 15
        self._grpc_data_channel_test(True)

    def _grpc_data_channel_test(self, time_based_flush=False):
        if False:
            for i in range(10):
                print('nop')
        if time_based_flush:
            data_servicer = data_plane.BeamFnDataServicer(data_buffer_time_limit_ms=100)
        else:
            data_servicer = data_plane.BeamFnDataServicer()
        worker_id = 'worker_0'
        data_channel_service = data_servicer.get_conn_by_worker_id(worker_id)
        server = grpc.server(thread_pool_executor.shared_unbounded_instance())
        beam_fn_api_pb2_grpc.add_BeamFnDataServicer_to_server(data_servicer, server)
        test_port = server.add_insecure_port('[::]:0')
        server.start()
        grpc_channel = grpc.insecure_channel('localhost:%s' % test_port)
        grpc_channel = grpc.intercept_channel(grpc_channel, WorkerIdInterceptor(worker_id))
        data_channel_stub = beam_fn_api_pb2_grpc.BeamFnDataStub(grpc_channel)
        if time_based_flush:
            data_channel_client = data_plane.GrpcClientDataChannel(data_channel_stub, data_buffer_time_limit_ms=100)
        else:
            data_channel_client = data_plane.GrpcClientDataChannel(data_channel_stub)
        try:
            self._data_channel_test(data_channel_service, data_channel_client, time_based_flush)
        finally:
            data_channel_client.close()
            data_channel_service.close()
            data_channel_client.wait()
            data_channel_service.wait()

    def test_in_memory_data_channel(self):
        if False:
            while True:
                i = 10
        channel = data_plane.InMemoryDataChannel()
        self._data_channel_test(channel, channel.inverse())

    def _data_channel_test(self, server, client, time_based_flush=False):
        if False:
            while True:
                i = 10
        self._data_channel_test_one_direction(server, client, time_based_flush)
        self._data_channel_test_one_direction(client, server, time_based_flush)

    def _data_channel_test_one_direction(self, from_channel, to_channel, time_based_flush):
        if False:
            return 10
        transform_1 = '1'
        transform_2 = '2'
        stream01 = from_channel.output_stream('0', transform_1)
        stream01.write(b'abc')
        if not time_based_flush:
            stream01.close()
        self.assertEqual(list(itertools.islice(to_channel.input_elements('0', [transform_1]), 1)), [beam_fn_api_pb2.Elements.Data(instruction_id='0', transform_id=transform_1, data=b'abc')])
        stream11 = from_channel.output_stream('1', transform_1)
        stream11.write(b'abc')
        stream21 = from_channel.output_stream('2', transform_1)
        stream21.write(b'def')
        if not time_based_flush:
            stream11.close()
        self.assertEqual(list(itertools.islice(to_channel.input_elements('1', [transform_1]), 1)), [beam_fn_api_pb2.Elements.Data(instruction_id='1', transform_id=transform_1, data=b'abc')])
        if time_based_flush:
            time.sleep(0.1)
        else:
            stream21.close()
        stream22 = from_channel.output_stream('2', transform_2)
        stream22.write(b'ghi')
        if not time_based_flush:
            stream22.close()
        self.assertEqual(list(itertools.islice(to_channel.input_elements('2', [transform_1, transform_2]), 2)), [beam_fn_api_pb2.Elements.Data(instruction_id='2', transform_id=transform_1, data=b'def'), beam_fn_api_pb2.Elements.Data(instruction_id='2', transform_id=transform_2, data=b'ghi')])
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()