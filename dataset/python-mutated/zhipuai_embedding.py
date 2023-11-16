from core.model_providers.error import LLMBadRequestError
from core.model_providers.providers.base import BaseModelProvider
from core.model_providers.models.embedding.base import BaseEmbedding
from core.third_party.langchain.embeddings.zhipuai_embedding import ZhipuAIEmbeddings

class ZhipuAIEmbedding(BaseEmbedding):

    def __init__(self, model_provider: BaseModelProvider, name: str):
        if False:
            print('Hello World!')
        credentials = model_provider.get_model_credentials(model_name=name, model_type=self.type)
        client = ZhipuAIEmbeddings(model=name, **credentials)
        super().__init__(model_provider, client, name)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if False:
            return 10
        return LLMBadRequestError(f'ZhipuAI embedding: {str(ex)}')