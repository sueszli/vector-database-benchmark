import sys
from logging import Handler

class FTStdErrStreamHandler(Handler):

    def flush(self):
        if False:
            i = 10
            return i + 15
        '\n        Override Flush behaviour - we keep half of the configured capacity\n        otherwise, we have moments with "empty" logs.\n        '
        self.acquire()
        try:
            sys.stderr.flush()
        finally:
            self.release()

    def emit(self, record):
        if False:
            return 10
        try:
            msg = self.format(record)
            sys.stderr.write(msg + '\n')
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)