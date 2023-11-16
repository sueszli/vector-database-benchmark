import json
from json import JSONDecodeError
from typing import Type
from core.helper import encrypter
from core.model_providers.models.base import BaseProviderModel
from core.model_providers.models.entity.model_params import ModelKwargsRules, KwargRule, ModelType, ModelMode
from core.model_providers.models.llm.tongyi_model import TongyiModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from core.third_party.langchain.llms.tongyi_llm import EnhanceTongyi
from models.provider import ProviderType

class TongyiProvider(BaseModelProvider):

    @property
    def provider_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the name of a provider.\n        '
        return 'tongyi'

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if False:
            while True:
                i = 10
        if model_type == ModelType.TEXT_GENERATION:
            return [{'id': 'qwen-turbo', 'name': 'qwen-turbo', 'mode': ModelMode.COMPLETION.value}, {'id': 'qwen-plus', 'name': 'qwen-plus', 'mode': ModelMode.COMPLETION.value}]
        else:
            return []

    def _get_text_generation_model_mode(self, model_name) -> str:
        if False:
            i = 10
            return i + 15
        return ModelMode.COMPLETION.value

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        if False:
            return 10
        '\n        Returns the model class.\n\n        :param model_type:\n        :return:\n        '
        if model_type == ModelType.TEXT_GENERATION:
            model_class = TongyiModel
        else:
            raise NotImplementedError
        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        if False:
            while True:
                i = 10
        '\n        get model parameter rules.\n\n        :param model_name:\n        :param model_type:\n        :return:\n        '
        model_max_tokens = {'qwen-turbo': 6000, 'qwen-plus': 6000}
        return ModelKwargsRules(temperature=KwargRule[float](min=0.01, max=1, default=1, precision=2), top_p=KwargRule[float](min=0.01, max=0.99, default=0.5, precision=2), presence_penalty=KwargRule[float](enabled=False), frequency_penalty=KwargRule[float](enabled=False), max_tokens=KwargRule[int](enabled=False, max=model_max_tokens.get(model_name)))

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        if False:
            print('Hello World!')
        '\n        Validates the given credentials.\n        '
        if 'dashscope_api_key' not in credentials:
            raise CredentialsValidateFailedError('Dashscope API Key must be provided.')
        try:
            credential_kwargs = {'dashscope_api_key': credentials['dashscope_api_key']}
            llm = EnhanceTongyi(model_name='qwen-turbo', max_retries=1, **credential_kwargs)
            llm('ping')
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        if False:
            i = 10
            return i + 15
        credentials['dashscope_api_key'] = encrypter.encrypt_token(tenant_id, credentials['dashscope_api_key'])
        return credentials

    def get_provider_credentials(self, obfuscated: bool=False) -> dict:
        if False:
            print('Hello World!')
        if self.provider.provider_type == ProviderType.CUSTOM.value:
            try:
                credentials = json.loads(self.provider.encrypted_config)
            except JSONDecodeError:
                credentials = {'dashscope_api_key': None}
            if credentials['dashscope_api_key']:
                credentials['dashscope_api_key'] = encrypter.decrypt_token(self.provider.tenant_id, credentials['dashscope_api_key'])
                if obfuscated:
                    credentials['dashscope_api_key'] = encrypter.obfuscated_token(credentials['dashscope_api_key'])
            return credentials
        return {}

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        if False:
            return 10
        '\n        check model credentials valid.\n\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        '
        return

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        encrypt model credentials for save.\n\n        :param tenant_id:\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        :return:\n        '
        return {}

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool=False) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        get credentials for llm use.\n\n        :param model_name:\n        :param model_type:\n        :param obfuscated:\n        :return:\n        '
        return self.get_provider_credentials(obfuscated)