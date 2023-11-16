from unittest.mock import MagicMock, Mock
from click import Option
from celery import Celery
app = Celery(set_as_current=False)
app.config_from_object('t.integration.test_worker_config')

class PurgeMock:

    def queue_purge(self, queue):
        if False:
            print('Hello World!')
        return 0

class ConnMock:
    default_channel = PurgeMock()
    channel_errors = KeyError
mock = Mock()
mock.__enter__ = Mock(return_value=ConnMock())
mock.__exit__ = Mock(return_value=False)
app.connection_for_write = MagicMock(return_value=mock)
ini_option = Option(('--ini', '-i'), help='Paste ini configuration file.')
ini_var_option = Option(('--ini-var',), help='Comma separated list of key=value to pass to ini.')
app.user_options['preload'].add(ini_option)
app.user_options['preload'].add(ini_var_option)