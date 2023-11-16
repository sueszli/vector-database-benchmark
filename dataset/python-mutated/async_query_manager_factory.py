from flask import Flask
from superset.async_events.async_query_manager import AsyncQueryManager
from superset.utils.class_utils import load_class_from_name

class AsyncQueryManagerFactory:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._async_query_manager: AsyncQueryManager = None

    def init_app(self, app: Flask) -> None:
        if False:
            print('Hello World!')
        self._async_query_manager = load_class_from_name(app.config['GLOBAL_ASYNC_QUERY_MANAGER_CLASS'])()
        self._async_query_manager.init_app(app)

    def instance(self) -> AsyncQueryManager:
        if False:
            print('Hello World!')
        return self._async_query_manager