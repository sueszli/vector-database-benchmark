"""Worker <-> Worker Sync at startup (Bootstep)."""
from celery import bootsteps
from celery.utils.log import get_logger
from .events import Events
__all__ = ('Mingle',)
logger = get_logger(__name__)
(debug, info, exception) = (logger.debug, logger.info, logger.exception)

class Mingle(bootsteps.StartStopStep):
    """Bootstep syncing state with neighbor workers.

    At startup, or upon consumer restart, this will:

    - Sync logical clocks.
    - Sync revoked tasks.

    """
    label = 'Mingle'
    requires = (Events,)
    compatible_transports = {'amqp', 'redis'}

    def __init__(self, c, without_mingle=False, **kwargs):
        if False:
            while True:
                i = 10
        self.enabled = not without_mingle and self.compatible_transport(c.app)
        super().__init__(c, without_mingle=without_mingle, **kwargs)

    def compatible_transport(self, app):
        if False:
            while True:
                i = 10
        with app.connection_for_read() as conn:
            return conn.transport.driver_type in self.compatible_transports

    def start(self, c):
        if False:
            for i in range(10):
                print('nop')
        self.sync(c)

    def sync(self, c):
        if False:
            i = 10
            return i + 15
        info('mingle: searching for neighbors')
        replies = self.send_hello(c)
        if replies:
            info('mingle: sync with %s nodes', len([reply for (reply, value) in replies.items() if value]))
            [self.on_node_reply(c, nodename, reply) for (nodename, reply) in replies.items() if reply]
            info('mingle: sync complete')
        else:
            info('mingle: all alone')

    def send_hello(self, c):
        if False:
            return 10
        inspect = c.app.control.inspect(timeout=1.0, connection=c.connection)
        our_revoked = c.controller.state.revoked
        replies = inspect.hello(c.hostname, our_revoked._data) or {}
        replies.pop(c.hostname, None)
        return replies

    def on_node_reply(self, c, nodename, reply):
        if False:
            print('Hello World!')
        debug('mingle: processing reply from %s', nodename)
        try:
            self.sync_with_node(c, **reply)
        except MemoryError:
            raise
        except Exception as exc:
            exception('mingle: sync with %s failed: %r', nodename, exc)

    def sync_with_node(self, c, clock=None, revoked=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.on_clock_event(c, clock)
        self.on_revoked_received(c, revoked)

    def on_clock_event(self, c, clock):
        if False:
            for i in range(10):
                print('nop')
        c.app.clock.adjust(clock) if clock else c.app.clock.forward()

    def on_revoked_received(self, c, revoked):
        if False:
            print('Hello World!')
        if revoked:
            c.controller.state.revoked.update(revoked)