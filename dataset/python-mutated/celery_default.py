from .celery_base import CeleryBaseService
__all__ = ['CeleryDefaultService']

class CeleryDefaultService(CeleryBaseService):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['queue'] = 'celery'
        super().__init__(**kwargs)

    def start_other(self):
        if False:
            return 10
        from terminal.startup import CeleryTerminal
        celery_terminal = CeleryTerminal()
        celery_terminal.start_heartbeat_thread()