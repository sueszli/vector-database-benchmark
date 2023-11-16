from __future__ import absolute_import
import sys
import six
from gunicorn.workers.sync import SyncWorker
__all__ = ['EventletSyncWorker']

class EventletSyncWorker(SyncWorker):
    """
    Custom sync worker for gunicorn which works with eventlet monkey patching.

    This worker class fixes "AssertionError: do not call blocking functions from
    the mainloop" and some other issues on SIGINT / SIGTERM signal.

    The actual issue happens in "time.sleep" call in "handle_quit" method -
    https://github.com/benoitc/gunicorn/blob/master/gunicorn/workers/base.py#L166
    which results in the assertion failure here -
    https://github.com/simplegeo/eventlet/blob/master/eventlet/greenthread.py#L27
    """

    def handle_quit(self, sig, frame):
        if False:
            return 10
        try:
            return super(EventletSyncWorker, self).handle_quit(sig=sig, frame=frame)
        except AssertionError as e:
            msg = six.text_type(e)
            if 'do not call blocking functions from the mainloop' in msg:
                sys.exit(0)
            raise e