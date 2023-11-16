import json
from json import JSONDecodeError
from typing import Type
from langchain.schema import HumanMessage
from core.helper import encrypter
from core.model_providers.models.base import BaseProviderModel
from core.model_providers.models.entity.model_params import ModelKwargsRules, KwargRule, ModelType, ModelMode
from core.model_providers.models.llm.baichuan_model import BaichuanModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from core.third_party.langchain.llms.baichuan_llm import BaichuanChatLLM
from models.provider import ProviderType

class BaichuanProvider(BaseModelProvider):

    @property
    def provider_name(self):
        if False:
            print('Hello World!')
        '\n        Returns the name of a provider.\n        '
        return 'baichuan'

    def _get_text_generation_model_mode(self, model_name) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ModelMode.CHAT.value

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if False:
            for i in range(10):
                print('nop')
        if model_type == ModelType.TEXT_GENERATION:
            return [{'id': 'baichuan2-53b', 'name': 'Baichuan2-53B', 'mode': ModelMode.CHAT.value}]
        else:
            return []

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the model class.\n\n        :param model_type:\n        :return:\n        '
        if model_type == ModelType.TEXT_GENERATION:
            model_class = BaichuanModel
        else:
            raise NotImplementedError
        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        if False:
            return 10
        '\n        get model parameter rules.\n\n        :param model_name:\n        :param model_type:\n        :return:\n        '
        return ModelKwargsRules(temperature=KwargRule[float](min=0, max=1, default=0.3, precision=2), top_p=KwargRule[float](min=0, max=0.99, default=0.85, precision=2), presence_penalty=KwargRule[float](enabled=False), frequency_penalty=KwargRule[float](enabled=False), max_tokens=KwargRule[int](enabled=False))

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validates the given credentials.\n        '
        if 'api_key' not in credentials:
            raise CredentialsValidateFailedError('Baichuan api_key must be provided.')
        if 'secret_key' not in credentials:
            raise CredentialsValidateFailedError('Baichuan secret_key must be provided.')
        try:
            credential_kwargs = {'api_key': credentials['api_key'], 'secret_key': credentials['secret_key']}
            llm = BaichuanChatLLM(temperature=0, **credential_kwargs)
            llm([HumanMessage(content='ping')])
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        if False:
            i = 10
            return i + 15
        credentials['api_key'] = encrypter.encrypt_token(tenant_id, credentials['api_key'])
        credentials['secret_key'] = encrypter.encrypt_token(tenant_id, credentials['secret_key'])
        return credentials

    def get_provider_credentials(self, obfuscated: bool=False) -> dict:
        if False:
            print('Hello World!')
        if self.provider.provider_type == ProviderType.CUSTOM.value:
            try:
                credentials = json.loads(self.provider.encrypted_config)
            except JSONDecodeError:
                credentials = {'api_key': None, 'secret_key': None}
            if credentials['api_key']:
                credentials['api_key'] = encrypter.decrypt_token(self.provider.tenant_id, credentials['api_key'])
                if obfuscated:
                    credentials['api_key'] = encrypter.obfuscated_token(credentials['api_key'])
            if credentials['secret_key']:
                credentials['secret_key'] = encrypter.decrypt_token(self.provider.tenant_id, credentials['secret_key'])
                if obfuscated:
                    credentials['secret_key'] = encrypter.obfuscated_token(credentials['secret_key'])
            return credentials
        else:
            return {}

    def should_deduct_quota(self):
        if False:
            return 10
        return True

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        check model credentials valid.\n\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        '
        return

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        encrypt model credentials for save.\n\n        :param tenant_id:\n        :param model_name:\n        :param model_type:\n        :param credentials:\n        :return:\n        '
        return {}

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool=False) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        get credentials for llm use.\n\n        :param model_name:\n        :param model_type:\n        :param obfuscated:\n        :return:\n        '
        return self.get_provider_credentials(obfuscated)