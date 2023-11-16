from typing import Optional
from core.tool.provider.base import BaseToolProvider
from core.tool.provider.errors import ToolValidateFailedError
from core.tool.serpapi_wrapper import OptimizedSerpAPIWrapper
from models.tool import ToolProviderName

class SerpAPIToolProvider(BaseToolProvider):

    def get_provider_name(self) -> ToolProviderName:
        if False:
            print('Hello World!')
        '\n        Returns the name of the provider.\n\n        :return:\n        '
        return ToolProviderName.SERPAPI

    def get_credentials(self, obfuscated: bool=False) -> Optional[dict]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the credentials for SerpAPI as a dictionary.\n\n        :param obfuscated: obfuscate credentials if True\n        :return:\n        '
        tool_provider = self.get_provider(must_enabled=True)
        if not tool_provider:
            return None
        credentials = tool_provider.credentials
        if not credentials:
            return None
        if credentials.get('api_key'):
            credentials['api_key'] = self.decrypt_token(credentials.get('api_key'), obfuscated)
        return credentials

    def credentials_to_func_kwargs(self) -> Optional[dict]:
        if False:
            while True:
                i = 10
        '\n        Returns the credentials function kwargs as a dictionary.\n\n        :return:\n        '
        credentials = self.get_credentials()
        if not credentials:
            return None
        return {'serpapi_api_key': credentials.get('api_key')}

    def credentials_validate(self, credentials: dict):
        if False:
            i = 10
            return i + 15
        '\n        Validates the given credentials.\n\n        :param credentials:\n        :return:\n        '
        if 'api_key' not in credentials or not credentials.get('api_key'):
            raise ToolValidateFailedError('SerpAPI api_key is required.')
        api_key = credentials.get('api_key')
        try:
            OptimizedSerpAPIWrapper(serpapi_api_key=api_key).run(query='test')
        except Exception as e:
            raise ToolValidateFailedError('SerpAPI api_key is invalid. {}'.format(e))

    def encrypt_credentials(self, credentials: dict) -> Optional[dict]:
        if False:
            print('Hello World!')
        '\n        Encrypts the given credentials.\n\n        :param credentials:\n        :return:\n        '
        credentials['api_key'] = self.encrypt_token(credentials.get('api_key'))
        return credentials