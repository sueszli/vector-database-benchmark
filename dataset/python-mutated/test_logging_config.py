from __future__ import annotations
import contextlib
import importlib
import logging
import os
import pathlib
import sys
import tempfile
from unittest.mock import patch
import pytest
from airflow.configuration import conf
from tests.test_utils.config import conf_vars
SETTINGS_FILE_VALID = "\nLOGGING_CONFIG = {\n    'version': 1,\n    'disable_existing_loggers': False,\n    'formatters': {\n        'airflow.task': {\n            'format': '[%%(asctime)s] {{%%(filename)s:%%(lineno)d}} %%(levelname)s - %%(message)s'\n        },\n    },\n    'handlers': {\n        'console': {\n            'class': 'logging.StreamHandler',\n            'formatter': 'airflow.task',\n            'stream': 'ext://sys.stdout'\n        },\n        'task': {\n            'class': 'logging.StreamHandler',\n            'formatter': 'airflow.task',\n            'stream': 'ext://sys.stdout'\n        },\n    },\n    'loggers': {\n        'airflow.task': {\n            'handlers': ['task'],\n            'level': 'INFO',\n            'propagate': False,\n        },\n    }\n}\n"
SETTINGS_FILE_INVALID = "\nLOGGING_CONFIG = {\n    'version': 1,\n    'disable_existing_loggers': False,\n    'formatters': {\n        'airflow.task': {\n            'format': '[%%(asctime)s] {{%%(filename)s:%%(lineno)d}} %%(levelname)s - %%(message)s'\n        },\n    },\n    'handlers': {\n        'console': {\n            'class': 'logging.StreamHandler',\n            'formatter': 'airflow.task',\n            'stream': 'ext://sys.stdout'\n        }\n    },\n    'loggers': {\n        'airflow': {\n            'handlers': ['file.handler'], # this handler does not exists\n            'level': 'INFO',\n            'propagate': False\n        }\n    }\n}\n"
SETTINGS_FILE_EMPTY = '\n# Other settings here\n'
SETTINGS_DEFAULT_NAME = 'custom_airflow_local_settings'

def reset_logging():
    if False:
        return 10
    'Reset Logging'
    manager = logging.root.manager
    manager.disabled = logging.NOTSET
    airflow_loggers = [logger for (logger_name, logger) in manager.loggerDict.items() if logger_name.startswith('airflow')]
    for logger in airflow_loggers:
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)

@contextlib.contextmanager
def settings_context(content, directory=None, name='LOGGING_CONFIG'):
    if False:
        return 10
    '\n    Sets a settings file and puts it in the Python classpath\n\n    :param content:\n          The content of the settings file\n    :param directory: the directory\n    :param name: str\n    '
    initial_logging_config = os.environ.get('AIRFLOW__LOGGING__LOGGING_CONFIG_CLASS', '')
    try:
        settings_root = tempfile.mkdtemp()
        filename = f'{SETTINGS_DEFAULT_NAME}.py'
        if directory:
            dir_path = os.path.join(settings_root, directory)
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            basedir = settings_root
            for part in directory.split('/'):
                open(os.path.join(basedir, '__init__.py'), 'w').close()
                basedir = os.path.join(basedir, part)
            open(os.path.join(basedir, '__init__.py'), 'w').close()
            module = directory.replace('/', '.') + '.' + SETTINGS_DEFAULT_NAME + '.' + name
            settings_file = os.path.join(dir_path, filename)
        else:
            module = SETTINGS_DEFAULT_NAME + '.' + name
            settings_file = os.path.join(settings_root, filename)
        with open(settings_file, 'w') as handle:
            handle.writelines(content)
        sys.path.append(settings_root)
        os.environ['AIRFLOW__LOGGING__LOGGING_CONFIG_CLASS'] = module
        yield settings_file
    finally:
        os.environ['AIRFLOW__LOGGING__LOGGING_CONFIG_CLASS'] = initial_logging_config
        sys.path.remove(settings_root)

class TestLoggingSettings:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.old_modules = dict(sys.modules)

    def teardown_method(self):
        if False:
            return 10
        from airflow.config_templates import airflow_local_settings
        from airflow.logging_config import configure_logging
        for mod in list(sys.modules):
            if mod not in self.old_modules:
                del sys.modules[mod]
        reset_logging()
        importlib.reload(airflow_local_settings)
        configure_logging()

    def test_loading_invalid_local_settings(self):
        if False:
            while True:
                i = 10
        from airflow.logging_config import configure_logging, log
        with settings_context(SETTINGS_FILE_INVALID):
            with patch.object(log, 'error') as mock_info:
                with pytest.raises(ValueError):
                    configure_logging()
                mock_info.assert_called_once_with('Unable to load the config, contains a configuration error.')

    def test_loading_valid_complex_local_settings(self):
        if False:
            while True:
                i = 10
        module_structure = 'etc.airflow.config'
        dir_structure = module_structure.replace('.', '/')
        with settings_context(SETTINGS_FILE_VALID, dir_structure):
            from airflow.logging_config import configure_logging, log
            with patch.object(log, 'info') as mock_info:
                configure_logging()
                mock_info.assert_called_once_with('Successfully imported user-defined logging config from %s', f'etc.airflow.config.{SETTINGS_DEFAULT_NAME}.LOGGING_CONFIG')

    def test_loading_valid_local_settings(self):
        if False:
            return 10
        with settings_context(SETTINGS_FILE_VALID):
            from airflow.logging_config import configure_logging, log
            with patch.object(log, 'info') as mock_info:
                configure_logging()
                mock_info.assert_called_once_with('Successfully imported user-defined logging config from %s', f'{SETTINGS_DEFAULT_NAME}.LOGGING_CONFIG')

    def test_loading_no_local_settings(self):
        if False:
            for i in range(10):
                print('nop')
        with settings_context(SETTINGS_FILE_EMPTY):
            from airflow.logging_config import configure_logging
            with pytest.raises(ImportError):
                configure_logging()

    def test_when_the_config_key_does_not_exists(self):
        if False:
            i = 10
            return i + 15
        from airflow import logging_config
        with conf_vars({('logging', 'logging_config_class'): None}):
            with patch.object(logging_config.log, 'debug') as mock_debug:
                logging_config.configure_logging()
                mock_debug.assert_any_call('Could not find key logging_config_class in config')

    def test_loading_local_settings_without_logging_config(self):
        if False:
            while True:
                i = 10
        from airflow.logging_config import configure_logging, log
        with patch.object(log, 'debug') as mock_info:
            configure_logging()
            mock_info.assert_called_once_with('Unable to load custom logging, using default config instead')

    def test_1_9_config(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow.logging_config import configure_logging
        with conf_vars({('logging', 'task_log_reader'): 'file.task'}):
            with pytest.warns(DeprecationWarning, match='file.task'):
                configure_logging()
            assert conf.get('logging', 'task_log_reader') == 'task'

    def test_loading_remote_logging_with_wasb_handler(self):
        if False:
            i = 10
            return i + 15
        'Test if logging can be configured successfully for Azure Blob Storage'
        from airflow.config_templates import airflow_local_settings
        from airflow.logging_config import configure_logging
        from airflow.utils.log.wasb_task_handler import WasbTaskHandler
        with conf_vars({('logging', 'remote_logging'): 'True', ('logging', 'remote_log_conn_id'): 'some_wasb', ('logging', 'remote_base_log_folder'): 'wasb://some-folder'}):
            importlib.reload(airflow_local_settings)
            configure_logging()
        logger = logging.getLogger('airflow.task')
        assert isinstance(logger.handlers[0], WasbTaskHandler)

    @pytest.mark.parametrize('remote_base_log_folder, log_group_arn', [('cloudwatch://arn:aws:logs:aaaa:bbbbb:log-group:ccccc', 'arn:aws:logs:aaaa:bbbbb:log-group:ccccc'), ('cloudwatch://arn:aws:logs:aaaa:bbbbb:log-group:aws/ccccc', 'arn:aws:logs:aaaa:bbbbb:log-group:aws/ccccc'), ('cloudwatch://arn:aws:logs:aaaa:bbbbb:log-group:/aws/ecs/ccccc', 'arn:aws:logs:aaaa:bbbbb:log-group:/aws/ecs/ccccc')])
    def test_log_group_arns_remote_logging_with_cloudwatch_handler(self, remote_base_log_folder, log_group_arn):
        if False:
            for i in range(10):
                print('nop')
        'Test if the correct ARNs are configured for Cloudwatch'
        from airflow.config_templates import airflow_local_settings
        from airflow.logging_config import configure_logging
        with conf_vars({('logging', 'remote_logging'): 'True', ('logging', 'remote_log_conn_id'): 'some_cloudwatch', ('logging', 'remote_base_log_folder'): remote_base_log_folder}):
            importlib.reload(airflow_local_settings)
            configure_logging()
            assert airflow_local_settings.DEFAULT_LOGGING_CONFIG['handlers']['task']['log_group_arn'] == log_group_arn

    def test_loading_remote_logging_with_kwargs(self):
        if False:
            while True:
                i = 10
        'Test if logging can be configured successfully with kwargs'
        from airflow.config_templates import airflow_local_settings
        from airflow.logging_config import configure_logging
        from airflow.utils.log.s3_task_handler import S3TaskHandler
        with conf_vars({('logging', 'remote_logging'): 'True', ('logging', 'remote_log_conn_id'): 'some_s3', ('logging', 'remote_base_log_folder'): 's3://some-folder', ('logging', 'remote_task_handler_kwargs'): '{"delete_local_copy": true}'}):
            importlib.reload(airflow_local_settings)
            configure_logging()
        logger = logging.getLogger('airflow.task')
        assert isinstance(logger.handlers[0], S3TaskHandler)
        assert getattr(logger.handlers[0], 'delete_local_copy') is True