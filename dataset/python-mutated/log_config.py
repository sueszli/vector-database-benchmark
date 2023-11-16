from __future__ import annotations
from copy import deepcopy
from airflow.config_templates.airflow_local_settings import DEFAULT_LOGGING_CONFIG
LOGGING_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)
LOGGING_CONFIG['root']['level'] = 'INFO'

def logger(level):
    if False:
        for i in range(10):
            print('nop')
    return {'handlers': ['console', 'task'], 'level': level, 'propagate': True}
levels = {'openlineage': 'DEBUG', 'great_expectations': 'INFO', 'airflow.lineage': 'DEBUG', 'botocore': 'INFO'}
for (el, lvl) in levels.items():
    LOGGING_CONFIG['loggers'][el] = logger(lvl)