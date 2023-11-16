import json
from typing import Type
from langchain.embeddings import LocalAIEmbeddings
from langchain.schema import HumanMessage
from core.helper import encrypter
from core.model_providers.models.embedding.localai_embedding import LocalAIEmbedding
from core.model_providers.models.entity.model_params import ModelKwargsRules, ModelType, KwargRule, ModelMode
from core.model_providers.models.llm.localai_model import LocalAIModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from core.model_providers.models.base import BaseProviderModel
from core.third_party.langchain.llms.chat_open_ai import EnhanceChatOpenAI
from core.third_party.langchain.llms.open_ai import EnhanceOpenAI
from models.provider import ProviderType

class LocalAIProvider(BaseModelProvider):

    @property
    def provider_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the name of a provider.\n        '
        return 'localai'

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if False:
            for i in range(10):
                print('nop')
        return []

    def _get_text_generation_model_mode(self, model_name) -> str:
        if False:
            print('Hello World!')
        credentials = self.get_model_credentials(model_name, ModelType.TEXT_GENERATION)
        if credentials['completion_type'] == 'chat_completion':
            return ModelMode.CHAT.value
        else:
            return ModelMode.COMPLETION.value

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        if False:
            print('Hello World!')
        '\n        Returns the model class.\n\n        :param model_type:\n        :return:\n        '
        if model_type == ModelType.TEXT_GENERATION:
            model_class = LocalAIModel
        elif model_type == ModelType.EMBEDDINGS:
            model_class = LocalAIEmbedding
        else:
            raise NotImplementedError
        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        if False:
            while True:
                i = 10
        '\n        get model parameter rules.\n\n        :param model_name:\n        :param model_type:\n        :return:\n        '
        return ModelKwargsRules(temperature=KwargRule[float](min=0, max=2, default=0.7, precision=2), top_p=KwargRule[float](min=0, max=1, default=1, precision=2), max_tokens=KwargRule[int](min=10, max=4097, default=16, precision=0))

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        check model credentials valid.\n\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        '
        if 'server_url' not in credentials:
            raise CredentialsValidateFailedError('LocalAI Server URL must be provided.')
        try:
            if model_type == ModelType.EMBEDDINGS:
                model = LocalAIEmbeddings(model=model_name, openai_api_key='1', openai_api_base=credentials['server_url'])
                model.embed_query('ping')
            else:
                if 'completion_type' not in credentials or credentials['completion_type'] not in ['completion', 'chat_completion']:
                    raise CredentialsValidateFailedError('LocalAI Completion Type must be provided.')
                if credentials['completion_type'] == 'chat_completion':
                    model = EnhanceChatOpenAI(model_name=model_name, openai_api_key='1', openai_api_base=credentials['server_url'] + '/v1', max_tokens=10, request_timeout=60)
                    model([HumanMessage(content='ping')])
                else:
                    model = EnhanceOpenAI(model_name=model_name, openai_api_key='1', openai_api_base=credentials['server_url'] + '/v1', max_tokens=10, request_timeout=60)
                    model('ping')
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        if False:
            while True:
                i = 10
        '\n        encrypt model credentials for save.\n\n        :param tenant_id:\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        :return:\n        '
        credentials['server_url'] = encrypter.encrypt_token(tenant_id, credentials['server_url'])
        return credentials

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool=False) -> dict:
        if False:
            while True:
                i = 10
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
            print('Hello World!')
        return

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {}

    def get_provider_credentials(self, obfuscated: bool=False) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {}