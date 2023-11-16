from .base import BaseService
from ..hands import *
__all__ = ['GunicornService']

class GunicornService(BaseService):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.worker = kwargs['worker_gunicorn']
        super().__init__(**kwargs)

    @property
    def cmd(self):
        if False:
            while True:
                i = 10
        print('\n- Start Gunicorn WSGI HTTP Server')
        log_format = '%(h)s %(t)s %(L)ss "%(r)s" %(s)s %(b)s '
        bind = f'{HTTP_HOST}:{HTTP_PORT}'
        cmd = ['gunicorn', 'jumpserver.asgi:application', '-b', bind, '-k', 'uvicorn.workers.UvicornWorker', '-w', str(self.worker), '--max-requests', '10240', '--max-requests-jitter', '2048', '--access-logformat', log_format, '--access-logfile', '-']
        if DEBUG:
            cmd.append('--reload')
        return cmd

    @property
    def cwd(self):
        if False:
            i = 10
            return i + 15
        return APPS_DIR

    def start_other(self):
        if False:
            while True:
                i = 10
        from terminal.startup import CoreTerminal
        core_terminal = CoreTerminal()
        core_terminal.start_heartbeat_thread()