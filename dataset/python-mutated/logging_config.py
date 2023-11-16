from __future__ import annotations
import logging
import warnings
from logging.config import dictConfig
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
from airflow.utils.module_loading import import_string
log = logging.getLogger(__name__)

def configure_logging():
    if False:
        i = 10
        return i + 15
    'Configure & Validate Airflow Logging.'
    logging_class_path = ''
    try:
        logging_class_path = conf.get('logging', 'logging_config_class')
    except AirflowConfigException:
        log.debug('Could not find key logging_config_class in config')
    if logging_class_path:
        try:
            logging_config = import_string(logging_class_path)
            if not isinstance(logging_config, dict):
                raise ValueError('Logging Config should be of dict type')
            log.info('Successfully imported user-defined logging config from %s', logging_class_path)
        except Exception as err:
            raise ImportError(f'Unable to load custom logging from {logging_class_path} due to {err}')
    else:
        logging_class_path = 'airflow.config_templates.airflow_local_settings.DEFAULT_LOGGING_CONFIG'
        logging_config = import_string(logging_class_path)
        log.debug('Unable to load custom logging, using default config instead')
    try:
        if 'filters' in logging_config and 'mask_secrets' in logging_config['filters']:
            task_handler_config = logging_config['handlers']['task']
            task_handler_config.setdefault('filters', [])
            if 'mask_secrets' not in task_handler_config['filters']:
                task_handler_config['filters'].append('mask_secrets')
        dictConfig(logging_config)
    except (ValueError, KeyError) as e:
        log.error('Unable to load the config, contains a configuration error.')
        raise e
    validate_logging_config(logging_config)
    return logging_class_path

def validate_logging_config(logging_config):
    if False:
        i = 10
        return i + 15
    'Validate the provided Logging Config.'
    task_log_reader = conf.get('logging', 'task_log_reader')
    logger = logging.getLogger('airflow.task')

    def _get_handler(name):
        if False:
            while True:
                i = 10
        return next((h for h in logger.handlers if h.name == name), None)
    if _get_handler(task_log_reader) is None:
        if task_log_reader == 'file.task' and _get_handler('task'):
            warnings.warn(f'task_log_reader setting in [logging] has a deprecated value of {task_log_reader!r}, but no handler with this name was found. Please update your config to use task. Running config has been adjusted to match', DeprecationWarning)
            conf.set('logging', 'task_log_reader', 'task')
        else:
            raise AirflowConfigException(f"Configured task_log_reader {task_log_reader!r} was not a handler of the 'airflow.task' logger.")