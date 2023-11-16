import decimal
import logging
import openai
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from core.model_providers.error import LLMBadRequestError, LLMAuthorizationError, LLMRateLimitError, LLMAPIUnavailableError, LLMAPIConnectionError
from core.model_providers.models.embedding.base import BaseEmbedding
from core.model_providers.providers.base import BaseModelProvider
AZURE_OPENAI_API_VERSION = '2023-07-01-preview'

class AzureOpenAIEmbedding(BaseEmbedding):

    def __init__(self, model_provider: BaseModelProvider, name: str):
        if False:
            for i in range(10):
                print('nop')
        self.credentials = model_provider.get_model_credentials(model_name=name, model_type=self.type)
        client = OpenAIEmbeddings(deployment=name, openai_api_type='azure', openai_api_version=AZURE_OPENAI_API_VERSION, chunk_size=16, max_retries=1, openai_api_key=self.credentials.get('openai_api_key'), openai_api_base=self.credentials.get('openai_api_base'))
        super().__init__(model_provider, client, name)

    @property
    def base_model_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        get base model name (not deployment)\n        \n        :return: str\n        '
        return self.credentials.get('base_model_name')

    def get_num_tokens(self, text: str) -> int:
        if False:
            while True:
                i = 10
        '\n        get num tokens of text.\n\n        :param text:\n        :return:\n        '
        if len(text) == 0:
            return 0
        enc = tiktoken.encoding_for_model(self.credentials.get('base_model_name'))
        tokenized_text = enc.encode(text)
        return len(tokenized_text)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if False:
            while True:
                i = 10
        if isinstance(ex, openai.error.InvalidRequestError):
            logging.warning('Invalid request to Azure OpenAI API.')
            return LLMBadRequestError(str(ex))
        elif isinstance(ex, openai.error.APIConnectionError):
            logging.warning('Failed to connect to Azure OpenAI API.')
            return LLMAPIConnectionError(ex.__class__.__name__ + ':' + str(ex))
        elif isinstance(ex, (openai.error.APIError, openai.error.ServiceUnavailableError, openai.error.Timeout)):
            logging.warning('Azure OpenAI service unavailable.')
            return LLMAPIUnavailableError(ex.__class__.__name__ + ':' + str(ex))
        elif isinstance(ex, openai.error.RateLimitError):
            return LLMRateLimitError('Azure ' + str(ex))
        elif isinstance(ex, openai.error.AuthenticationError):
            return LLMAuthorizationError('Azure ' + str(ex))
        elif isinstance(ex, openai.error.OpenAIError):
            return LLMBadRequestError('Azure ' + ex.__class__.__name__ + ':' + str(ex))
        else:
            return ex