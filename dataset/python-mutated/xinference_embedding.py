from core.model_providers.error import LLMBadRequestError
from core.model_providers.providers.base import BaseModelProvider
from core.model_providers.models.embedding.base import BaseEmbedding
from core.third_party.langchain.embeddings.xinference_embedding import XinferenceEmbeddings

class XinferenceEmbedding(BaseEmbedding):

    def __init__(self, model_provider: BaseModelProvider, name: str):
        if False:
            for i in range(10):
                print('nop')
        credentials = model_provider.get_model_credentials(model_name=name, model_type=self.type)
        client = XinferenceEmbeddings(server_url=credentials['server_url'], model_uid=credentials['model_uid'])
        super().__init__(model_provider, client, name)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if False:
            i = 10
            return i + 15
        return LLMBadRequestError(f'Xinference embedding: {str(ex)}')