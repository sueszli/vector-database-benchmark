"""Consumer Broker Connection Bootstep."""
from kombu.common import ignore_errors
from celery import bootsteps
from celery.utils.log import get_logger
__all__ = ('Connection',)
logger = get_logger(__name__)
info = logger.info

class Connection(bootsteps.StartStopStep):
    """Service managing the consumer broker connection."""

    def __init__(self, c, **kwargs):
        if False:
            while True:
                i = 10
        c.connection = None
        super().__init__(c, **kwargs)

    def start(self, c):
        if False:
            while True:
                i = 10
        c.connection = c.connect()
        info('Connected to %s', c.connection.as_uri())

    def shutdown(self, c):
        if False:
            i = 10
            return i + 15
        (connection, c.connection) = (c.connection, None)
        if connection:
            ignore_errors(connection, connection.close)

    def info(self, c):
        if False:
            print('Hello World!')
        params = 'N/A'
        if c.connection:
            params = c.connection.info()
            params.pop('password', None)
        return {'broker': params}