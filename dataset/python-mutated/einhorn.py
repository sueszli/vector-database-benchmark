"""Run a Gunicorn WSGI container under Einhorn.

[Einhorn] is a language/protocol-agnostic socket and worker manager. We're
using it elsewhere for Baseplate services (where something WSGI-specific
wouldn't work on Thrift-based services) and its graceful reload logic is more
friendly than Gunicorn's*. However, for non-gevent WSGI, we still need
something to parse HTTP and provide a WSGI container. Gunicorn is excellent at
this. This module adapts Gunicorn to work under Einhorn as a single worker
process.

To run a paste-based application (like r2) under Einhorn using Gunicorn as the
WSGI container, run this module. All of gunicorn's command line arguments are
supported, though some may be meaningless (like worker count) because Gunicorn
isn't managing workers.

    einhorn -n 4 -b 0.0.0.0:8080 python -m r2.lib.einhorn example.ini

[Einhorn]: https://github.com/stripe/einhorn

*: In particular, when told to gracefully reload, Gunicorn will gracefully
terminate all workers immediately and then replace them. Einhorn starts up a
new worker, waits for it to acknowledge it is up and running, and then reaps an
old worker.

"""
import os
import signal
import sys
from baseplate.server import einhorn
from gunicorn import util
from gunicorn.app.pasterapp import PasterApplication
from gunicorn.workers.sync import SyncWorker

class EinhornSyncWorker(SyncWorker):

    def __init__(self, cfg, app):
        if False:
            print('Hello World!')
        listener = einhorn.get_socket()
        super(EinhornSyncWorker, self).__init__(age=0, ppid=os.getppid(), sockets=[listener], app=app, timeout=None, cfg=cfg, log=cfg.logger_class(cfg))

    def init_signals(self):
        if False:
            i = 10
            return i + 15
        [signal.signal(s, signal.SIG_DFL) for s in self.SIGNALS]
        signal.signal(signal.SIGUSR2, self.start_graceful_shutdown)
        signal.siginterrupt(signal.SIGUSR2, False)

    def start_graceful_shutdown(self, signal_number, frame):
        if False:
            i = 10
            return i + 15
        self.alive = False

def run_gunicorn_worker():
    if False:
        return 10
    if not einhorn.is_worker():
        (print >> sys.stderr, 'This process does not appear to be running under Einhorn.')
        sys.exit(1)
    app = PasterApplication()
    util._setproctitle('worker [%s]' % app.cfg.proc_name)
    worker = EinhornSyncWorker(app.cfg, app)
    worker.init_process()
if __name__ == '__main__':
    run_gunicorn_worker()