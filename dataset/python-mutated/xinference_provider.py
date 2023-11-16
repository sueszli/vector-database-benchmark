import json
from typing import Type
import requests
from core.helper import encrypter
from core.model_providers.models.embedding.xinference_embedding import XinferenceEmbedding
from core.model_providers.models.entity.model_params import KwargRule, ModelKwargsRules, ModelType, ModelMode
from core.model_providers.models.llm.xinference_model import XinferenceModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from core.model_providers.models.base import BaseProviderModel
from core.third_party.langchain.embeddings.xinference_embedding import XinferenceEmbeddings
from core.third_party.langchain.llms.xinference_llm import XinferenceLLM
from models.provider import ProviderType

class XinferenceProvider(BaseModelProvider):

    @property
    def provider_name(self):
        if False:
            while True:
                i = 10
        '\n        Returns the name of a provider.\n        '
        return 'xinference'

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if False:
            while True:
                i = 10
        return []

    def _get_text_generation_model_mode(self, model_name) -> str:
        if False:
            print('Hello World!')
        return ModelMode.COMPLETION.value

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        if False:
            print('Hello World!')
        '\n        Returns the model class.\n\n        :param model_type:\n        :return:\n        '
        if model_type == ModelType.TEXT_GENERATION:
            model_class = XinferenceModel
        elif model_type == ModelType.EMBEDDINGS:
            model_class = XinferenceEmbedding
        else:
            raise NotImplementedError
        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        if False:
            while True:
                i = 10
        '\n        get model parameter rules.\n\n        :param model_name:\n        :param model_type:\n        :return:\n        '
        credentials = self.get_model_credentials(model_name, model_type)
        if credentials['model_format'] == 'ggmlv3' and credentials['model_handle_type'] == 'chatglm':
            return ModelKwargsRules(temperature=KwargRule[float](min=0.01, max=2, default=1, precision=2), top_p=KwargRule[float](min=0, max=1, default=0.7, precision=2), presence_penalty=KwargRule[float](enabled=False), frequency_penalty=KwargRule[float](enabled=False), max_tokens=KwargRule[int](min=10, max=4000, default=256, precision=0))
        elif credentials['model_format'] == 'ggmlv3':
            return ModelKwargsRules(temperature=KwargRule[float](min=0.01, max=2, default=1, precision=2), top_p=KwargRule[float](min=0, max=1, default=0.7, precision=2), presence_penalty=KwargRule[float](min=-2, max=2, default=0, precision=2), frequency_penalty=KwargRule[float](min=-2, max=2, default=0, precision=2), max_tokens=KwargRule[int](min=10, max=4000, default=256, precision=0))
        else:
            return ModelKwargsRules(temperature=KwargRule[float](min=0.01, max=2, default=1, precision=2), top_p=KwargRule[float](min=0, max=1, default=0.7, precision=2), presence_penalty=KwargRule[float](enabled=False), frequency_penalty=KwargRule[float](enabled=False), max_tokens=KwargRule[int](min=10, max=4000, default=256, precision=0))

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        if False:
            i = 10
            return i + 15
        '\n        check model credentials valid.\n\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        '
        if 'server_url' not in credentials:
            raise CredentialsValidateFailedError('Xinference Server URL must be provided.')
        if 'model_uid' not in credentials:
            raise CredentialsValidateFailedError('Xinference Model UID must be provided.')
        try:
            credential_kwargs = {'server_url': credentials['server_url'], 'model_uid': credentials['model_uid']}
            if model_type == ModelType.TEXT_GENERATION:
                llm = XinferenceLLM(**credential_kwargs)
                llm('ping')
            elif model_type == ModelType.EMBEDDINGS:
                embedding = XinferenceEmbeddings(**credential_kwargs)
                embedding.embed_query('ping')
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        encrypt model credentials for save.\n\n        :param tenant_id:\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        :return:\n        '
        if model_type == ModelType.TEXT_GENERATION:
            extra_credentials = cls._get_extra_credentials(credentials)
            credentials.update(extra_credentials)
        credentials['server_url'] = encrypter.encrypt_token(tenant_id, credentials['server_url'])
        return credentials

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool=False) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        get credentials for llm use.\n\n        :param model_name:\n        :param model_type:\n        :param obfuscated:\n        :return:\n        '
        if self.provider.provider_type != ProviderType.CUSTOM.value:
            raise NotImplementedError
        provider_model = self._get_provider_model(model_name, model_type)
        if not provider_model.encrypted_config:
            return {'server_url': None, 'model_uid': None}
        credentials = json.loads(provider_model.encrypted_config)
        if credentials['server_url']:
            credentials['server_url'] = encrypter.decrypt_token(self.provider.tenant_id, credentials['server_url'])
            if obfuscated:
                credentials['server_url'] = encrypter.obfuscated_token(credentials['server_url'])
        return credentials

    @classmethod
    def _get_extra_credentials(self, credentials: dict) -> dict:
        if False:
            return 10
        url = f"{credentials['server_url']}/v1/models/{credentials['model_uid']}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get the model description, detail: {response.json()['detail']}")
        desc = response.json()
        extra_credentials = {'model_format': desc['model_format']}
        if desc['model_format'] == 'ggmlv3' and 'chatglm' in desc['model_name']:
            extra_credentials['model_handle_type'] = 'chatglm'
        elif 'generate' in desc['model_ability']:
            extra_credentials['model_handle_type'] = 'generate'
        elif 'chat' in desc['model_ability']:
            extra_credentials['model_handle_type'] = 'chat'
        else:
            raise NotImplementedError(f'Model handle type not supported.')
        return extra_credentials

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        if False:
            return 10
        return

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        if False:
            while True:
                i = 10
        return {}

    def get_provider_credentials(self, obfuscated: bool=False) -> dict:
        if False:
            i = 10
            return i + 15
        return {}