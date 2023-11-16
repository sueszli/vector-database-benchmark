from typing import Dict
from litestar import Litestar, get
from litestar.config.app import AppConfig
from litestar.di import Provide
from litestar.plugins import InitPluginProtocol

@get('/', sync_to_thread=False)
def route_handler(name: str) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    return {'hello': name}

def get_name() -> str:
    if False:
        print('Hello World!')
    return 'world'

class MyPlugin(InitPluginProtocol):

    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        if False:
            for i in range(10):
                print('nop')
        app_config.dependencies['name'] = Provide(get_name, sync_to_thread=False)
        app_config.route_handlers.append(route_handler)
        return app_config
app = Litestar(plugins=[MyPlugin()])