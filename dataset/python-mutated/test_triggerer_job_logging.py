from __future__ import annotations
import importlib
import logging
import warnings
import pytest
from airflow.config_templates import airflow_local_settings
from airflow.jobs import triggerer_job_runner
from airflow.logging_config import configure_logging
from airflow.providers.amazon.aws.log.s3_task_handler import S3TaskHandler
from airflow.utils.log.file_task_handler import FileTaskHandler
from airflow.utils.log.logging_mixin import RedirectStdHandler
from airflow.utils.log.trigger_handler import DropTriggerLogsFilter, TriggererHandlerWrapper
from tests.test_utils.config import conf_vars

def non_pytest_handlers(val):
    if False:
        return 10
    return [h for h in val if 'pytest' not in h.__module__]

def assert_handlers(logger, *classes):
    if False:
        while True:
            i = 10
    handlers = non_pytest_handlers(logger.handlers)
    assert [x.__class__ for x in handlers] == list(classes or [])
    return handlers

def clear_logger_handlers(log):
    if False:
        print('Hello World!')
    for h in log.handlers[:]:
        if 'pytest' not in h.__module__:
            log.removeHandler(h)

@pytest.fixture(autouse=True)
def reload_triggerer_job():
    if False:
        i = 10
        return i + 15
    importlib.reload(triggerer_job_runner)

def test_configure_trigger_log_handler_file():
    if False:
        i = 10
        return i + 15
    '\n    root logger: RedirectStdHandler\n    task: FTH\n    result: wrap\n\n    '
    root_logger = logging.getLogger()
    clear_logger_handlers(root_logger)
    configure_logging()
    assert_handlers(root_logger, RedirectStdHandler)
    task_logger = logging.getLogger('airflow.task')
    task_handlers = assert_handlers(task_logger, FileTaskHandler)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    triggerer_job_runner.configure_trigger_log_handler()
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is True
    root_handlers = assert_handlers(root_logger, RedirectStdHandler, TriggererHandlerWrapper)
    assert root_handlers[1].base_handler == task_handlers[0]
    assert root_handlers[0].filters[1].__class__ == DropTriggerLogsFilter
    assert root_handlers[1].filters == []
    assert root_handlers[1].base_handler.__class__ == FileTaskHandler

def test_configure_trigger_log_handler_s3():
    if False:
        i = 10
        return i + 15
    '\n    root logger: RedirectStdHandler\n    task: S3TH\n    result: wrap\n    '
    with conf_vars({('logging', 'remote_logging'): 'True', ('logging', 'remote_log_conn_id'): 'some_aws', ('logging', 'remote_base_log_folder'): 's3://some-folder'}):
        importlib.reload(airflow_local_settings)
        configure_logging()
    root_logger = logging.getLogger()
    assert_handlers(root_logger, RedirectStdHandler)
    task_logger = logging.getLogger('airflow.task')
    task_handlers = assert_handlers(task_logger, S3TaskHandler)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    triggerer_job_runner.configure_trigger_log_handler()
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is True
    handlers = assert_handlers(root_logger, RedirectStdHandler, TriggererHandlerWrapper)
    assert handlers[1].base_handler == task_handlers[0]
    assert handlers[0].filters[1].__class__ == DropTriggerLogsFilter
    assert handlers[1].filters == []
    assert handlers[1].base_handler.__class__ == S3TaskHandler

class OldFileTaskHandler(FileTaskHandler):
    """Handler that hasn't been updated to support triggerer"""

    def _read(self, ti, try_number, metadata=None):
        if False:
            i = 10
            return i + 15
        super()._read(self, ti, try_number, metadata)
non_file_task_handler = {'version': 1, 'handlers': {'task': {'class': 'logging.Handler'}}, 'loggers': {'airflow.task': {'handlers': ['task']}}}
old_file_task_handler = {'version': 1, 'handlers': {'task': {'class': 'tests.jobs.test_triggerer_job_logging.OldFileTaskHandler', 'base_log_folder': 'hello'}}, 'loggers': {'airflow.task': {'handlers': ['task']}}}
not_supported_message = ['Handler OldFileTaskHandler does not support individual trigger logging. Please check the release notes for your provider to see if a newer version supports individual trigger logging.', 'Could not find log handler suitable for individual trigger logging.']
not_found_message = ['Could not find log handler suitable for individual trigger logging.']

@pytest.mark.parametrize('cfg, cls, msg', [('old_file_task_handler', OldFileTaskHandler, not_supported_message), ('non_file_task_handler', logging.Handler, not_found_message)])
def test_configure_trigger_log_handler_not_file_task_handler(cfg, cls, msg):
    if False:
        for i in range(10):
            print('nop')
    "\n    No root handler configured.\n    When non FileTaskHandler is configured, don't modify.\n    When an incompatible subclass of FileTaskHandler is configured, don't modify.\n    "
    root_logger = logging.getLogger()
    clear_logger_handlers(root_logger)
    with conf_vars({('logging', 'logging_config_class'): f'tests.jobs.test_triggerer_job_logging.{cfg}'}):
        importlib.reload(airflow_local_settings)
        configure_logging()
    assert_handlers(root_logger)
    task_logger = logging.getLogger('airflow.task')
    assert_handlers(task_logger, cls)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    with warnings.catch_warnings(record=True) as captured:
        triggerer_job_runner.configure_trigger_log_handler()
    assert [x.message.args[0] for x in captured] == msg
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    assert_handlers(root_logger)
fallback_task = {'version': 1, 'handlers': {'task': {'class': 'airflow.providers.amazon.aws.log.s3_task_handler.S3TaskHandler', 'base_log_folder': '~/abc', 's3_log_folder': 's3://abc', 'filename_template': 'blah'}}, 'loggers': {'airflow.task': {'handlers': ['task']}}}

def test_configure_trigger_log_handler_fallback_task():
    if False:
        for i in range(10):
            print('nop')
    '\n    root: no handler\n    task: FTH\n    result: wrap\n    '
    with conf_vars({('logging', 'logging_config_class'): 'tests.jobs.test_triggerer_job_logging.fallback_task'}):
        importlib.reload(airflow_local_settings)
        configure_logging()
    task_logger = logging.getLogger('airflow.task')
    assert_handlers(task_logger, S3TaskHandler)
    root_logger = logging.getLogger()
    assert_handlers(root_logger)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    triggerer_job_runner.configure_trigger_log_handler()
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is True
    handlers = assert_handlers(root_logger, TriggererHandlerWrapper)
    assert handlers[0].base_handler == task_logger.handlers[0]
    assert handlers[0].filters == []
root_has_task_handler = {'version': 1, 'handlers': {'task': {'class': 'logging.Handler'}, 'trigger': {'class': 'airflow.utils.log.file_task_handler.FileTaskHandler', 'base_log_folder': 'blah'}}, 'loggers': {'airflow.task': {'handlers': ['task']}, '': {'handlers': ['trigger']}}}

def test_configure_trigger_log_handler_root_has_task_handler():
    if False:
        while True:
            i = 10
    '\n    root logger: single handler that supports triggerer\n    result: wrap\n    '
    with conf_vars({('logging', 'logging_config_class'): 'tests.jobs.test_triggerer_job_logging.root_has_task_handler'}):
        configure_logging()
    task_logger = logging.getLogger('airflow.task')
    assert_handlers(task_logger, logging.Handler)
    root_logger = logging.getLogger()
    assert_handlers(root_logger, FileTaskHandler)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    triggerer_job_runner.configure_trigger_log_handler()
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is True
    handlers = assert_handlers(root_logger, TriggererHandlerWrapper)
    assert handlers[0].filters == []
    assert handlers[0].base_handler.__class__ == FileTaskHandler
root_not_file_task = {'version': 1, 'handlers': {'task': {'class': 'airflow.providers.amazon.aws.log.s3_task_handler.S3TaskHandler', 'base_log_folder': '~/abc', 's3_log_folder': 's3://abc', 'filename_template': 'blah'}, 'trigger': {'class': 'logging.Handler'}}, 'loggers': {'airflow.task': {'handlers': ['task']}, '': {'handlers': ['trigger']}}}

def test_configure_trigger_log_handler_root_not_file_task():
    if False:
        while True:
            i = 10
    "\n    root: A handler that doesn't support trigger or inherit FileTaskHandler\n    task: Supports triggerer\n    Result:\n        * wrap and use the task logger handler\n        * other root handlers filter trigger logging\n    "
    with conf_vars({('logging', 'logging_config_class'): 'tests.jobs.test_triggerer_job_logging.root_not_file_task'}):
        configure_logging()
    task_logger = logging.getLogger('airflow.task')
    assert_handlers(task_logger, S3TaskHandler)
    root_logger = logging.getLogger()
    assert_handlers(root_logger, logging.Handler)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    with warnings.catch_warnings(record=True) as captured:
        triggerer_job_runner.configure_trigger_log_handler()
    assert captured == []
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is True
    handlers = assert_handlers(root_logger, logging.Handler, TriggererHandlerWrapper)
    assert handlers[0].filters[0].__class__ == DropTriggerLogsFilter
    assert handlers[1].filters == []
    assert handlers[1].base_handler.__class__ == S3TaskHandler
root_logger_old_file_task = {'version': 1, 'handlers': {'task': {'class': 'airflow.providers.amazon.aws.log.s3_task_handler.S3TaskHandler', 'base_log_folder': '~/abc', 's3_log_folder': 's3://abc', 'filename_template': 'blah'}, 'trigger': {'class': 'tests.jobs.test_triggerer_job_logging.OldFileTaskHandler', 'base_log_folder': 'abc'}}, 'loggers': {'airflow.task': {'handlers': ['task']}, '': {'handlers': ['trigger']}}}

def test_configure_trigger_log_handler_root_old_file_task():
    if False:
        print('Hello World!')
    "\n    Root logger handler: An older subclass of FileTaskHandler that doesn't support triggerer\n    Task logger handler: Supports triggerer\n    Result:\n        * wrap and use the task logger handler\n        * other root handlers filter trigger logging\n    "
    with conf_vars({('logging', 'logging_config_class'): 'tests.jobs.test_triggerer_job_logging.root_logger_old_file_task'}):
        configure_logging()
    assert_handlers(logging.getLogger('airflow.task'), S3TaskHandler)
    root_logger = logging.getLogger()
    assert_handlers(root_logger, OldFileTaskHandler)
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is False
    with warnings.catch_warnings(record=True) as captured:
        triggerer_job_runner.configure_trigger_log_handler()
    assert [x.message.args[0] for x in captured] == ['Handler OldFileTaskHandler does not support individual trigger logging. Please check the release notes for your provider to see if a newer version supports individual trigger logging.']
    assert triggerer_job_runner.HANDLER_SUPPORTS_TRIGGERER is True
    handlers = assert_handlers(root_logger, OldFileTaskHandler, TriggererHandlerWrapper)
    assert handlers[0].filters[0].__class__ == DropTriggerLogsFilter
    assert handlers[1].filters == []
    assert handlers[1].base_handler.__class__ == S3TaskHandler