"""Client Interceptor to inject worker_id"""
import collections
import os
from typing import Optional
import grpc

class _ClientCallDetails(collections.namedtuple('_ClientCallDetails', ('method', 'timeout', 'metadata', 'credentials')), grpc.ClientCallDetails):
    pass

class WorkerIdInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.StreamStreamClientInterceptor):
    _worker_id = os.environ.get('WORKER_ID')

    def __init__(self, worker_id=None):
        if False:
            while True:
                i = 10
        if worker_id:
            self._worker_id = worker_id

    def intercept_unary_unary(self, continuation, client_call_details, request):
        if False:
            print('Hello World!')
        return self._intercept(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        if False:
            for i in range(10):
                print('nop')
        return self._intercept(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request):
        if False:
            while True:
                i = 10
        return self._intercept(continuation, client_call_details, request)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        if False:
            print('Hello World!')
        return self._intercept(continuation, client_call_details, request_iterator)

    def _intercept(self, continuation, client_call_details, request):
        if False:
            while True:
                i = 10
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        if 'worker_id' in metadata:
            raise RuntimeError('Header metadata already has a worker_id.')
        metadata.append(('worker_id', self._worker_id))
        new_client_details = _ClientCallDetails(client_call_details.method, client_call_details.timeout, metadata, client_call_details.credentials)
        return continuation(new_client_details, request)