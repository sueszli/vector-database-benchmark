from __future__ import absolute_import
import tempfile
import six
import mock
from oslo_config.cfg import ConfigFilesNotFoundError
from st2common import service_setup
from st2common.services import coordination
from st2common.transport.bootstrap_utils import register_exchanges
from st2common.transport.bootstrap_utils import QUEUES
from st2tests.base import CleanFilesTestCase
from st2tests import config
__all__ = ['ServiceSetupTestCase']
MOCK_LOGGING_CONFIG_INVALID_LOG_LEVEL = '\n[loggers]\nkeys=root\n\n[handlers]\nkeys=consoleHandler\n\n[formatters]\nkeys=simpleConsoleFormatter\n\n[logger_root]\nlevel=invalid_log_level\nhandlers=consoleHandler\n\n[handler_consoleHandler]\nclass=StreamHandler\nlevel=DEBUG\nformatter=simpleConsoleFormatter\nargs=(sys.stdout,)\n\n[formatter_simpleConsoleFormatter]\nclass=st2common.logging.formatters.ConsoleLogFormatter\nformat=%(asctime)s %(levelname)s [-] %(message)s\ndatefmt=\n'.strip()
MOCK_LOGGING_CONFIG_VALID = '\n[loggers]\nkeys=root\n\n[handlers]\nkeys=consoleHandler\n\n[formatters]\nkeys=simpleConsoleFormatter\n\n[logger_root]\nlevel=DEBUG\nhandlers=consoleHandler\n\n[handler_consoleHandler]\nclass=StreamHandler\nlevel=DEBUG\nformatter=simpleConsoleFormatter\nargs=(sys.stdout,)\n\n[formatter_simpleConsoleFormatter]\nclass=st2common.logging.formatters.ConsoleLogFormatter\nformat=%(asctime)s %(levelname)s [-] %(message)s\ndatefmt=\n'.strip()
MOCK_DEFAULT_CONFIG_FILE_PATH = '/etc/st2/st2.conf-test-patched'

def mock_get_logging_config_path():
    if False:
        print('Hello World!')
    return ''

class ServiceSetupTestCase(CleanFilesTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ServiceSetupTestCase, self).setUp()
        config.USE_DEFAULT_CONFIG_FILES = False

    def tearDown(self):
        if False:
            return 10
        super(ServiceSetupTestCase, self).tearDown()
        config.USE_DEFAULT_CONFIG_FILES = False

    def test_no_logging_config_found(self):
        if False:
            print('Hello World!')
        config.get_logging_config_path = mock_get_logging_config_path
        if six.PY3:
            expected_msg = '.*KeyError:.*'
        else:
            expected_msg = 'No section: .*'
        self.assertRaisesRegexp(Exception, expected_msg, service_setup.setup, service='api', config=config, setup_db=False, register_mq_exchanges=False, register_signal_handlers=False, register_internal_trigger_types=False, run_migrations=False)

    def test_invalid_log_level_friendly_error_message(self):
        if False:
            return 10
        (_, mock_logging_config_path) = tempfile.mkstemp()
        self.to_delete_files.append(mock_logging_config_path)
        with open(mock_logging_config_path, 'w') as fp:
            fp.write(MOCK_LOGGING_CONFIG_INVALID_LOG_LEVEL)

        def mock_get_logging_config_path():
            if False:
                print('Hello World!')
            return mock_logging_config_path
        config.get_logging_config_path = mock_get_logging_config_path
        if six.PY3:
            expected_msg = "ValueError: Unknown level: 'invalid_log_level'"
            exc_type = ValueError
        else:
            expected_msg = 'Invalid log level selected. Log level names need to be all uppercase'
            exc_type = KeyError
        self.assertRaisesRegexp(exc_type, expected_msg, service_setup.setup, service='api', config=config, setup_db=False, register_mq_exchanges=False, register_signal_handlers=False, register_internal_trigger_types=False, run_migrations=False)

    @mock.patch('kombu.Queue.declare')
    def test_register_exchanges_predeclare_queues(self, mock_declare):
        if False:
            return 10
        self.assertEqual(mock_declare.call_count, 0)
        register_exchanges()
        self.assertEqual(mock_declare.call_count, len(QUEUES))

    @mock.patch('st2tests.config.DEFAULT_CONFIG_FILE_PATH', MOCK_DEFAULT_CONFIG_FILE_PATH)
    def test_service_setup_default_st2_conf_config_is_used(self):
        if False:
            print('Hello World!')
        config.USE_DEFAULT_CONFIG_FILES = True
        (_, mock_logging_config_path) = tempfile.mkstemp()
        self.to_delete_files.append(mock_logging_config_path)
        with open(mock_logging_config_path, 'w') as fp:
            fp.write(MOCK_LOGGING_CONFIG_VALID)

        def mock_get_logging_config_path():
            if False:
                print('Hello World!')
            return mock_logging_config_path
        config.get_logging_config_path = mock_get_logging_config_path
        expected_msg = 'Failed to find some config files: %s' % MOCK_DEFAULT_CONFIG_FILE_PATH
        self.assertRaisesRegexp(ConfigFilesNotFoundError, expected_msg, service_setup.setup, service='api', config=config, config_args=['--debug'], setup_db=False, register_mq_exchanges=False, register_signal_handlers=False, register_internal_trigger_types=False, run_migrations=False, register_runners=False)
        config_file_path = '/etc/st2/config.override.test'
        expected_msg = 'Failed to find some config files: %s' % config_file_path
        self.assertRaisesRegexp(ConfigFilesNotFoundError, expected_msg, service_setup.setup, service='api', config=config, config_args=['--config-file', config_file_path], setup_db=False, register_mq_exchanges=False, register_signal_handlers=False, register_internal_trigger_types=False, run_migrations=False, register_runners=False)

    def test_deregister_service_when_service_registry_enabled(self):
        if False:
            print('Hello World!')
        service = 'api'
        service_setup.register_service_in_service_registry(service, capabilities={'hostname': '', 'pid': ''})
        coordinator = coordination.get_coordinator(start_heart=True)
        members = coordinator.get_members(service.encode('utf-8'))
        self.assertEqual(len(list(members.get())), 1)
        service_setup.deregister_service(service)
        self.assertEqual(len(list(members.get())), 0)

    def test_deregister_service_when_service_registry_disables(self):
        if False:
            for i in range(10):
                print('nop')
        service = 'api'
        try:
            service_setup.deregister_service(service)
        except:
            assert False, 'service_setup.deregister_service raised exception'
        assert True