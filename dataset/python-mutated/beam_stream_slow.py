from apache_beam.coders.coder_impl import create_InputStream, create_OutputStream
from pyflink.fn_execution.stream_slow import InputStream
from pyflink.fn_execution.utils.operation_utils import PeriodicThread

class BeamInputStream(InputStream):

    def __init__(self, input_stream: create_InputStream):
        if False:
            i = 10
            return i + 15
        super(BeamInputStream, self).__init__([])
        self._input_stream = input_stream

    def read(self, size):
        if False:
            print('Hello World!')
        return self._input_stream.read(size)

    def read_byte(self):
        if False:
            print('Hello World!')
        return self._input_stream.read_byte()

    def size(self):
        if False:
            print('Hello World!')
        return self._input_stream.size()

class BeamTimeBasedOutputStream(create_OutputStream):

    def __init__(self):
        if False:
            print('Hello World!')
        super(BeamTimeBasedOutputStream).__init__()
        self._flush_event = False
        self._periodic_flusher = PeriodicThread(1, self.notify_flush)
        self._periodic_flusher.daemon = True
        self._periodic_flusher.start()
        self._output_stream = None

    def write(self, b: bytes):
        if False:
            print('Hello World!')
        self._output_stream.write(b)

    def reset_output_stream(self, output_stream: create_OutputStream):
        if False:
            while True:
                i = 10
        self._output_stream = output_stream

    def notify_flush(self):
        if False:
            return 10
        self._flush_event = True

    def close(self):
        if False:
            while True:
                i = 10
        if self._periodic_flusher:
            self._periodic_flusher.cancel()
            self._periodic_flusher = None

    def maybe_flush(self):
        if False:
            print('Hello World!')
        if self._flush_event:
            self._output_stream.flush()
            self._flush_event = False
        else:
            self._output_stream.maybe_flush()