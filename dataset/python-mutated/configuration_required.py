"""`Configuration` provider required modifier example."""
from dependency_injector import containers, providers, errors

class ApiClient:

    def __init__(self, api_key: str, timeout: int):
        if False:
            print('Hello World!')
        self.api_key = api_key
        self.timeout = timeout

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    api_client_factory = providers.Factory(ApiClient, api_key=config.api.key.required(), timeout=config.api.timeout.required().as_int())
if __name__ == '__main__':
    container = Container()
    try:
        api_client = container.api_client_factory()
    except errors.Error:
        ...