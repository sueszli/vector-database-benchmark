import logging
import os
import twisted.logger
from synapse.logging.context import LoggingContextFilter
from synapse.synapse_rust import reset_logging_config

class ToTwistedHandler(logging.Handler):
    """logging handler which sends the logs to the twisted log"""
    tx_log = twisted.logger.Logger()

    def emit(self, record: logging.LogRecord) -> None:
        if False:
            i = 10
            return i + 15
        log_entry = self.format(record)
        log_level = record.levelname.lower().replace('warning', 'warn')
        self.tx_log.emit(twisted.logger.LogLevel.levelWithName(log_level), '{entry}', entry=log_entry)

def setup_logging() -> None:
    if False:
        return 10
    'Configure the python logging appropriately for the tests.\n\n    (Logs will end up in _trial_temp.)\n    '
    root_logger = logging.getLogger()
    log_format = '%(name)s - %(lineno)d - %(levelname)s - %(request)s - %(message)s'
    handler = ToTwistedHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    handler.addFilter(LoggingContextFilter())
    root_logger.addHandler(handler)
    log_level = os.environ.get('SYNAPSE_TEST_LOG_LEVEL', 'ERROR')
    root_logger.setLevel(log_level)
    if root_logger.isEnabledFor(logging.INFO):
        logging.getLogger('synapse.visibility.filtered_event_debug').setLevel(logging.DEBUG)
    reset_logging_config()