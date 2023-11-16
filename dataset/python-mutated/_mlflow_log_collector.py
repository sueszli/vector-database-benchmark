import os
import sys
import errno
import mlflow
import traceback
import tempfile

class OutputCollector(object):

    def __init__(self, stream, processor):
        if False:
            for i in range(10):
                print('nop')
        self._inner = stream
        self.processor = processor

    def write(self, buf):
        if False:
            for i in range(10):
                print('nop')
        self.processor(buf)
        self._inner.write(buf)

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        return getattr(self._inner, name)

class RedirectUserOutputStreams(object):

    def __init__(self, logger):
        if False:
            return 10
        self.logger = logger
        self.user_log_path = tempfile.mkstemp(suffix='_stdout_stderr.txt')[1]

    def __enter__(self):
        if False:
            return 10
        self.logger.debug('Redirecting user output to {0}'.format(self.user_log_path))
        self.user_log_fp = open(self.user_log_path, 'at+')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = OutputCollector(sys.stdout, self.user_log_fp.write)
        sys.stderr = OutputCollector(sys.stderr, self.user_log_fp.write)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        try:
            if exc_val:
                trace = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
                print(trace, file=sys.stderr)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            mlflow.log_artifact(self.user_log_path, 'user_logs')
            self.user_log_fp.close()
            self.logger.debug('User scope execution complete.')