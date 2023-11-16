from unittest import mock
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from after import ApiClient, Service

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    api_client = providers.Singleton(ApiClient, api_key=config.api_key, timeout=config.timeout)
    service = providers.Factory(Service, api_client=api_client)

@inject
def main(service: Service=Provide[Container.service]) -> None:
    if False:
        i = 10
        return i + 15
    ...
if __name__ == '__main__':
    container = Container()
    container.config.api_key.from_env('API_KEY', required=True)
    container.config.timeout.from_env('TIMEOUT', as_=int, default=5)
    container.wire(modules=[__name__])
    main()
    with container.api_client.override(mock.Mock()):
        main()