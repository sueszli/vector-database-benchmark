"""
This module contains helper classes for configuring logging for luigid and
workers via command line arguments and options from config files.
"""
import logging
import logging.config
import os.path
from luigi.configuration import get_config, LuigiConfigParser
from luigi.freezing import recursively_unfreeze
from configparser import NoSectionError

class BaseLogging:
    config = get_config()

    @classmethod
    def _section(cls, opts):
        if False:
            print('Hello World!')
        'Get logging settings from config file section "logging".'
        if isinstance(cls.config, LuigiConfigParser):
            return False
        try:
            logging_config = cls.config['logging']
        except (TypeError, KeyError, NoSectionError):
            return False
        logging.config.dictConfig(recursively_unfreeze(logging_config))
        return True

    @classmethod
    def setup(cls, opts=type('opts', (), {'background': None, 'logdir': None, 'logging_conf_file': None, 'log_level': 'DEBUG'})):
        if False:
            i = 10
            return i + 15
        'Setup logging via CLI params and config.'
        logger = logging.getLogger('luigi')
        if cls._configured:
            logger.info('logging already configured')
            return False
        cls._configured = True
        if cls.config.getboolean('core', 'no_configure_logging', False):
            logger.info('logging disabled in settings')
            return False
        configured = cls._cli(opts)
        if configured:
            logger = logging.getLogger('luigi')
            logger.info('logging configured via special settings')
            return True
        configured = cls._conf(opts)
        if configured:
            logger = logging.getLogger('luigi')
            logger.info('logging configured via *.conf file')
            return True
        configured = cls._section(opts)
        if configured:
            logger = logging.getLogger('luigi')
            logger.info('logging configured via config section')
            return True
        configured = cls._default(opts)
        if configured:
            logger = logging.getLogger('luigi')
            logger.info('logging configured by default settings')
        return configured

class DaemonLogging(BaseLogging):
    """Configure logging for luigid
    """
    _configured = False
    _log_format = '%(asctime)s %(name)s[%(process)s] %(levelname)s: %(message)s'

    @classmethod
    def _cli(cls, opts):
        if False:
            while True:
                i = 10
        "Setup logging via CLI options\n\n        If `--background` -- set INFO level for root logger.\n        If `--logdir` -- set logging with next params:\n            default Luigi's formatter,\n            INFO level,\n            output in logdir in `luigi-server.log` file\n        "
        if opts.background:
            logging.getLogger().setLevel(logging.INFO)
            return True
        if opts.logdir:
            logging.basicConfig(level=logging.INFO, format=cls._log_format, filename=os.path.join(opts.logdir, 'luigi-server.log'))
            return True
        return False

    @classmethod
    def _conf(cls, opts):
        if False:
            i = 10
            return i + 15
        'Setup logging via ini-file from logging_conf_file option.'
        logging_conf = cls.config.get('core', 'logging_conf_file', None)
        if logging_conf is None:
            return False
        if not os.path.exists(logging_conf):
            raise OSError('Error: Unable to locate specified logging configuration file!')
        logging.config.fileConfig(logging_conf)
        return True

    @classmethod
    def _default(cls, opts):
        if False:
            return 10
        'Setup default logger'
        logging.basicConfig(level=logging.INFO, format=cls._log_format)
        return True

class InterfaceLogging(BaseLogging):
    """Configure logging for worker"""
    _configured = False

    @classmethod
    def _cli(cls, opts):
        if False:
            while True:
                i = 10
        return False

    @classmethod
    def _conf(cls, opts):
        if False:
            while True:
                i = 10
        'Setup logging via ini-file from logging_conf_file option.'
        if not opts.logging_conf_file:
            return False
        if not os.path.exists(opts.logging_conf_file):
            raise OSError('Error: Unable to locate specified logging configuration file!')
        logging.config.fileConfig(opts.logging_conf_file, disable_existing_loggers=False)
        return True

    @classmethod
    def _default(cls, opts):
        if False:
            while True:
                i = 10
        'Setup default logger'
        level = getattr(logging, opts.log_level, logging.DEBUG)
        logger = logging.getLogger('luigi-interface')
        logger.setLevel(level)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return True