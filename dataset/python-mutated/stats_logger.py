from flask import Flask
from superset.stats_logger import BaseStatsLogger

class BaseStatsLoggerManager:

    def __init__(self) -> None:
        if False:
            return 10
        self._stats_logger = BaseStatsLogger()

    def init_app(self, app: Flask) -> None:
        if False:
            return 10
        self._stats_logger = app.config['STATS_LOGGER']

    @property
    def instance(self) -> BaseStatsLogger:
        if False:
            while True:
                i = 10
        return self._stats_logger