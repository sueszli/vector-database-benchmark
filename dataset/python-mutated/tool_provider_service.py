from typing import Optional
from core.tool.provider.base import BaseToolProvider
from core.tool.provider.serpapi_provider import SerpAPIToolProvider

class ToolProviderService:

    def __init__(self, tenant_id: str, provider_name: str):
        if False:
            print('Hello World!')
        self.provider = self._init_provider(tenant_id, provider_name)

    def _init_provider(self, tenant_id: str, provider_name: str) -> BaseToolProvider:
        if False:
            print('Hello World!')
        if provider_name == 'serpapi':
            return SerpAPIToolProvider(tenant_id)
        else:
            raise Exception('tool provider {} not found'.format(provider_name))

    def get_credentials(self, obfuscated: bool=False) -> Optional[dict]:
        if False:
            return 10
        '\n        Returns the credentials for Tool as a dictionary.\n\n        :param obfuscated:\n        :return:\n        '
        return self.provider.get_credentials(obfuscated)

    def credentials_validate(self, credentials: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validates the given credentials.\n\n        :param credentials:\n        :raises: ValidateFailedError\n        '
        return self.provider.credentials_validate(credentials)

    def encrypt_credentials(self, credentials: dict):
        if False:
            return 10
        '\n        Encrypts the given credentials.\n\n        :param credentials:\n        :return:\n        '
        return self.provider.encrypt_credentials(credentials)