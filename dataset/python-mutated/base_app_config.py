import logging
from typing import Optional
from embedchain.config.base_config import BaseConfig
from embedchain.helper.json_serializable import JSONSerializable
from embedchain.vectordb.base import BaseVectorDB

class BaseAppConfig(BaseConfig, JSONSerializable):
    """
    Parent config to initialize an instance of `App`.
    """

    def __init__(self, log_level: str='WARNING', db: Optional[BaseVectorDB]=None, id: Optional[str]=None, collect_metrics: bool=True, collection_name: Optional[str]=None):
        if False:
            while True:
                i = 10
        '\n        Initializes a configuration class instance for an App.\n        Most of the configuration is done in the `App` class itself.\n\n        :param log_level: Debug level [\'DEBUG\', \'INFO\', \'WARNING\', \'ERROR\', \'CRITICAL\'], defaults to "WARNING"\n        :type log_level: str, optional\n        :param db: A database class. It is recommended to set this directly in the `App` class, not this config,\n        defaults to None\n        :type db: Optional[BaseVectorDB], optional\n        :param id: ID of the app. Document metadata will have this id., defaults to None\n        :type id: Optional[str], optional\n        :param collect_metrics: Send anonymous telemetry to improve embedchain, defaults to True\n        :type collect_metrics: Optional[bool], optional\n        :param collection_name: Default collection name. It\'s recommended to use app.db.set_collection_name() instead,\n        defaults to None\n        :type collection_name: Optional[str], optional\n        '
        self._setup_logging(log_level)
        self.id = id
        self.collect_metrics = True if collect_metrics is True or collect_metrics is None else False
        self.collection_name = collection_name
        if db:
            self._db = db
            logging.warning('DEPRECATION WARNING: Please supply the database as the second parameter during app init. Such as `app(config=config, db=db)`.')
        if collection_name:
            logging.warning('DEPRECATION WARNING: Please supply the collection name to the database config.')
        return

    def _setup_logging(self, debug_level):
        if False:
            while True:
                i = 10
        level = logging.WARNING
        if debug_level is not None:
            level = getattr(logging, debug_level.upper(), None)
            if not isinstance(level, int):
                raise ValueError(f'Invalid log level: {debug_level}')
        logging.basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', level=level)
        self.logger = logging.getLogger(__name__)
        return