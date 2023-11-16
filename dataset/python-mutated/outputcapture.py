from io import StringIO
import sys
from robot.output import LOGGER
from robot.utils import console_decode, console_encode

class OutputCapturer:

    def __init__(self, library_import=False):
        if False:
            for i in range(10):
                print('nop')
        self._library_import = library_import
        self._python_out = PythonCapturer(stdout=True)
        self._python_err = PythonCapturer(stdout=False)

    def __enter__(self):
        if False:
            while True:
                i = 10
        if self._library_import:
            LOGGER.enable_library_import_logging()
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        if False:
            for i in range(10):
                print('nop')
        self._release_and_log()
        if self._library_import:
            LOGGER.disable_library_import_logging()
        return False

    def _release_and_log(self):
        if False:
            for i in range(10):
                print('nop')
        (stdout, stderr) = self._release()
        if stdout:
            LOGGER.log_output(stdout)
        if stderr:
            LOGGER.log_output(stderr)
            sys.__stderr__.write(console_encode(stderr, stream=sys.__stderr__))

    def _release(self):
        if False:
            i = 10
            return i + 15
        stdout = self._python_out.release()
        stderr = self._python_err.release()
        return (stdout, stderr)

class PythonCapturer:

    def __init__(self, stdout=True):
        if False:
            return 10
        if stdout:
            self._original = sys.stdout
            self._set_stream = self._set_stdout
        else:
            self._original = sys.stderr
            self._set_stream = self._set_stderr
        self._stream = StringIO()
        self._set_stream(self._stream)

    def _set_stdout(self, stream):
        if False:
            return 10
        sys.stdout = stream

    def _set_stderr(self, stream):
        if False:
            print('Hello World!')
        sys.stderr = stream

    def release(self):
        if False:
            return 10
        self._set_stream(self._original)
        try:
            return self._get_value(self._stream)
        finally:
            self._stream.close()
            self._avoid_at_exit_errors(self._stream)

    def _get_value(self, stream):
        if False:
            for i in range(10):
                print('nop')
        try:
            return console_decode(stream.getvalue())
        except UnicodeError:
            stream.buf = console_decode(stream.buf)
            stream.buflist = [console_decode(item) for item in stream.buflist]
            return stream.getvalue()

    def _avoid_at_exit_errors(self, stream):
        if False:
            print('Hello World!')
        stream.write = lambda s: None
        stream.flush = lambda : None