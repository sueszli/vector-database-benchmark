"""Test for WorkerIdInterceptor"""
import collections
import logging
import unittest
import grpc
from apache_beam.runners.worker.worker_id_interceptor import WorkerIdInterceptor

class _ClientCallDetails(collections.namedtuple('_ClientCallDetails', ('method', 'timeout', 'metadata', 'credentials')), grpc.ClientCallDetails):
    pass

class WorkerIdInterceptorTest(unittest.TestCase):

    def test_worker_id_insertion(self):
        if False:
            print('Hello World!')
        worker_id_key = 'worker_id'
        headers_holder = {}

        def continuation(client_details, request_iterator):
            if False:
                for i in range(10):
                    print('nop')
            headers_holder.update({worker_id_key: dict(client_details.metadata).get(worker_id_key)})
        WorkerIdInterceptor._worker_id = 'my_worker_id'
        WorkerIdInterceptor().intercept_stream_stream(continuation, _ClientCallDetails(None, None, None, None), [])
        self.assertEqual(headers_holder[worker_id_key], 'my_worker_id', 'worker_id_key not set')

    def test_failure_when_worker_id_exists(self):
        if False:
            for i in range(10):
                print('nop')
        worker_id_key = 'worker_id'
        headers_holder = {}

        def continuation(client_details, request_iterator):
            if False:
                i = 10
                return i + 15
            headers_holder.update({worker_id_key: dict(client_details.metadata).get(worker_id_key)})
        WorkerIdInterceptor._worker_id = 'my_worker_id'
        with self.assertRaises(RuntimeError):
            WorkerIdInterceptor().intercept_stream_stream(continuation, _ClientCallDetails(None, None, {'worker_id': '1'}, None), [])
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()