import sys
from metaflow.event_logger import NullEventLogger
from metaflow.sidecar import Message, MessageTypes

class DebugEventLogger(NullEventLogger):
    TYPE = 'debugLogger'

    @classmethod
    def get_worker(cls):
        if False:
            i = 10
            return i + 15
        return DebugEventLoggerSidecar

class DebugEventLoggerSidecar(object):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def process_message(self, msg):
        if False:
            i = 10
            return i + 15
        if msg.msg_type == MessageTypes.SHUTDOWN:
            print('Debug[shutdown]: got shutdown!', file=sys.stderr)
            self._shutdown()
        elif msg.msg_type == MessageTypes.BEST_EFFORT:
            print('Debug[best_effort]: %s' % str(msg.payload), file=sys.stderr)
        elif msg.msg_type == MessageTypes.MUST_SEND:
            print('Debug[must_send]: %s' % str(msg.payload), file=sys.stderr)

    def _shutdown(self):
        if False:
            return 10
        sys.stderr.flush()