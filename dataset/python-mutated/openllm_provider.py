import json
from typing import Type
from core.helper import encrypter
from core.model_providers.models.embedding.openllm_embedding import OpenLLMEmbedding
from core.model_providers.models.entity.model_params import KwargRule, ModelKwargsRules, ModelType, ModelMode
from core.model_providers.models.llm.openllm_model import OpenLLMModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from core.model_providers.models.base import BaseProviderModel
from core.third_party.langchain.embeddings.openllm_embedding import OpenLLMEmbeddings
from core.third_party.langchain.llms.openllm import OpenLLM
from models.provider import ProviderType

class OpenLLMProvider(BaseModelProvider):

    @property
    def provider_name(self):
        if False:
            print('Hello World!')
        '\n        Returns the name of a provider.\n        '
        return 'openllm'

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if False:
            return 10
        return []

    def _get_text_generation_model_mode(self, model_name) -> str:
        if False:
            i = 10
            return i + 15
        return ModelMode.COMPLETION.value

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        if False:
            print('Hello World!')
        '\n        Returns the model class.\n\n        :param model_type:\n        :return:\n        '
        if model_type == ModelType.TEXT_GENERATION:
            model_class = OpenLLMModel
        elif model_type == ModelType.EMBEDDINGS:
            model_class = OpenLLMEmbedding
        else:
            raise NotImplementedError
        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        if False:
            print('Hello World!')
        '\n        get model parameter rules.\n\n        :param model_name:\n        :param model_type:\n        :return:\n        '
        return ModelKwargsRules(temperature=KwargRule[float](min=0.01, max=2, default=1, precision=2), top_p=KwargRule[float](min=0, max=1, default=0.7, precision=2), presence_penalty=KwargRule[float](min=-2, max=2, default=0, precision=2), frequency_penalty=KwargRule[float](min=-2, max=2, default=0, precision=2), max_tokens=KwargRule[int](alias='max_new_tokens', min=10, max=4000, default=128, precision=0))

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        if False:
            i = 10
            return i + 15
        '\n        check model credentials valid.\n\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        '
        if 'server_url' not in credentials:
            raise CredentialsValidateFailedError('OpenLLM Server URL must be provided.')
        try:
            credential_kwargs = {'server_url': credentials['server_url']}
            if model_type == ModelType.TEXT_GENERATION:
                llm = OpenLLM(llm_kwargs={'max_new_tokens': 10}, **credential_kwargs)
                llm('ping')
            elif model_type == ModelType.EMBEDDINGS:
                embedding = OpenLLMEmbeddings(**credential_kwargs)
                embedding.embed_query('ping')
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        encrypt model credentials for save.\n\n        :param tenant_id:\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        :return:\n        '
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
            return {'server_url': None}
        credentials = json.loads(provider_model.encrypted_config)
        if credentials['server_url']:
            credentials['server_url'] = encrypter.decrypt_token(self.provider.tenant_id, credentials['server_url'])
            if obfuscated:
                credentials['server_url'] = encrypter.obfuscated_token(credentials['server_url'])
        return credentials

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        if False:
            while True:
                i = 10
        return

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {}

    def get_provider_credentials(self, obfuscated: bool=False) -> dict:
        if False:
            while True:
                i = 10
        return {}