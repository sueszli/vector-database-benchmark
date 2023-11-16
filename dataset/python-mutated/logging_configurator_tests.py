import logging
import unittest
from unittest.mock import MagicMock
from superset.utils.logging_configurator import LoggingConfigurator

class TestLoggingConfigurator(unittest.TestCase):

    def reset_logging(self):
        if False:
            for i in range(10):
                print('nop')
        logging.root.manager.loggerDict = {}
        logging.root.handlers = []

    def test_configurator_adding_handler(self):
        if False:
            print('Hello World!')

        class MyEventHandler(logging.Handler):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__(level=logging.DEBUG)
                self.received = False

            def handle(self, record):
                if False:
                    i = 10
                    return i + 15
                if hasattr(record, 'testattr'):
                    self.received = True

        class MyConfigurator(LoggingConfigurator):

            def __init__(self, handler):
                if False:
                    while True:
                        i = 10
                self.handler = handler

            def configure_logging(self, app_config, debug_mode):
                if False:
                    return 10
                super().configure_logging(app_config, debug_mode)
                logging.getLogger().addHandler(self.handler)
        self.reset_logging()
        handler = MyEventHandler()
        cfg = MyConfigurator(handler)
        cfg.configure_logging(MagicMock(), True)
        logging.info('test', extra={'testattr': 'foo'})
        self.assertTrue(handler.received)