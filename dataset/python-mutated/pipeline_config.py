from typing import Optional
from embedchain.helper.json_serializable import register_deserializable
from .apps.base_app_config import BaseAppConfig

@register_deserializable
class PipelineConfig(BaseAppConfig):
    """
    Config to initialize an embedchain custom `App` instance, with extra config options.
    """

    def __init__(self, log_level: str='WARNING', id: Optional[str]=None, name: Optional[str]=None, collect_metrics: Optional[bool]=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes a configuration class instance for an App. This is the simplest form of an embedchain app.\n        Most of the configuration is done in the `App` class itself.\n\n        :param log_level: Debug level [\'DEBUG\', \'INFO\', \'WARNING\', \'ERROR\', \'CRITICAL\'], defaults to "WARNING"\n        :type log_level: str, optional\n        :param id: ID of the app. Document metadata will have this id., defaults to None\n        :type id: Optional[str], optional\n        :param collect_metrics: Send anonymous telemetry to improve embedchain, defaults to True\n        :type collect_metrics: Optional[bool], optional\n        :param collection_name: Default collection name. It\'s recommended to use app.db.set_collection_name() instead,\n        defaults to None\n        :type collection_name: Optional[str], optional\n        '
        self._setup_logging(log_level)
        self.id = id
        self.name = name
        self.collect_metrics = collect_metrics