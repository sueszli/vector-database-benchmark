from core.model_providers.error import LLMBadRequestError
from core.model_providers.providers.base import BaseModelProvider
from core.third_party.langchain.embeddings.huggingface_hub_embedding import HuggingfaceHubEmbeddings
from core.model_providers.models.embedding.base import BaseEmbedding

class HuggingfaceEmbedding(BaseEmbedding):

    def __init__(self, model_provider: BaseModelProvider, name: str):
        if False:
            print('Hello World!')
        credentials = model_provider.get_model_credentials(model_name=name, model_type=self.type)
        client = HuggingfaceHubEmbeddings(model=name, **credentials)
        super().__init__(model_provider, client, name)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if False:
            for i in range(10):
                print('nop')
        return LLMBadRequestError(f'Huggingface embedding: {str(ex)}')