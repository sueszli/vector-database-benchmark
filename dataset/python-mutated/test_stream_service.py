from concurrent.futures import ThreadPoolExecutor
import grpc
from apache_beam.portability.api import beam_runner_api_pb2_grpc

class TestStreamServiceController(beam_runner_api_pb2_grpc.TestStreamServiceServicer):
    """A server that streams TestStreamPayload.Events from a single EventRequest.

  This server is used as a way for TestStreams to receive events from file.
  """

    def __init__(self, reader, endpoint=None, exception_handler=None):
        if False:
            for i in range(10):
                print('nop')
        self._server = grpc.server(ThreadPoolExecutor(max_workers=10))
        self._server_started = False
        self._server_stopped = False
        if endpoint:
            self.endpoint = endpoint
            self._server.add_insecure_port(self.endpoint)
        else:
            port = self._server.add_insecure_port('localhost:0')
            self.endpoint = 'localhost:{}'.format(port)
        beam_runner_api_pb2_grpc.add_TestStreamServiceServicer_to_server(self, self._server)
        self._reader = reader
        self._exception_handler = exception_handler
        if not self._exception_handler:
            self._exception_handler = lambda _: False

    def start(self):
        if False:
            while True:
                i = 10
        if self._server_started or self._server_stopped:
            return
        self._server_started = True
        self._server.start()

    def stop(self):
        if False:
            while True:
                i = 10
        if not self._server_started or self._server_stopped:
            return
        self._server_started = False
        self._server_stopped = True
        self._server.stop(0)
        if hasattr(self._server, 'wait_for_termination'):
            self._server.wait_for_termination()

    def Events(self, request, context):
        if False:
            return 10
        'Streams back all of the events from the streaming cache.'
        tags = [None if tag == 'None' else tag for tag in request.output_ids]
        try:
            reader = self._reader.read_multiple([('full', tag) for tag in tags])
            while True:
                e = next(reader)
                yield e
        except StopIteration:
            pass
        except Exception as e:
            if not self._exception_handler(e):
                raise e