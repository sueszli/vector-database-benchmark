from typing import Type
from core.model_providers.models.base import BaseProviderModel
from core.model_providers.models.entity.model_params import ModelType, ModelKwargsRules, ModelMode
from core.model_providers.models.llm.openai_model import OpenAIModel
from core.model_providers.providers.base import BaseModelProvider

class FakeModelProvider(BaseModelProvider):

    @property
    def provider_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'fake'

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if False:
            return 10
        return [{'id': 'test_model', 'name': 'Test Model', 'mode': 'completion'}]

    def _get_text_generation_model_mode(self, model_name) -> str:
        if False:
            while True:
                i = 10
        return ModelMode.COMPLETION.value

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        if False:
            for i in range(10):
                print('nop')
        return OpenAIModel

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        if False:
            return 10
        pass

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        if False:
            print('Hello World!')
        return credentials

    def get_provider_credentials(self, obfuscated: bool=False) -> dict:
        if False:
            return 10
        return {}

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        if False:
            i = 10
            return i + 15
        return credentials

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        if False:
            print('Hello World!')
        return ModelKwargsRules()

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool=False) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {}