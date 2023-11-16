import abc
import logging
from logging.handlers import TimedRotatingFileHandler
import flask.app
import flask.config
logger = logging.getLogger(__name__)

class LoggingConfigurator(abc.ABC):

    @abc.abstractmethod
    def configure_logging(self, app_config: flask.config.Config, debug_mode: bool) -> None:
        if False:
            return 10
        pass

class DefaultLoggingConfigurator(LoggingConfigurator):

    def configure_logging(self, app_config: flask.config.Config, debug_mode: bool) -> None:
        if False:
            print('Hello World!')
        if app_config['SILENCE_FAB']:
            logging.getLogger('flask_appbuilder').setLevel(logging.ERROR)
        superset_logger = logging.getLogger('superset')
        if debug_mode:
            superset_logger.setLevel(logging.DEBUG)
        else:
            superset_logger.addHandler(logging.StreamHandler())
            superset_logger.setLevel(logging.INFO)
        logging.getLogger('pyhive.presto').setLevel(logging.INFO)
        logging.basicConfig(format=app_config['LOG_FORMAT'])
        logging.getLogger().setLevel(app_config['LOG_LEVEL'])
        if app_config['ENABLE_TIME_ROTATE']:
            logging.getLogger().setLevel(app_config['TIME_ROTATE_LOG_LEVEL'])
            handler = TimedRotatingFileHandler(app_config['FILENAME'], when=app_config['ROLLOVER'], interval=app_config['INTERVAL'], backupCount=app_config['BACKUP_COUNT'])
            logging.getLogger().addHandler(handler)
        logger.info('logging was configured successfully')