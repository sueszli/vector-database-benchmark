from typing import List, Optional
from airbyte_cdk.destinations.vector_db_based.document_processor import Chunk
from airbyte_cdk.destinations.vector_db_based.embedder import Embedder
from destination_chroma.config import NoEmbeddingConfigModel

class NoEmbedder(Embedder):

    def __init__(self, config: NoEmbeddingConfigModel):
        if False:
            while True:
                i = 10
        super().__init__()

    def check(self) -> Optional[str]:
        if False:
            return 10
        return None

    def embed_chunks(self, chunks: List[Chunk]) -> List[None]:
        if False:
            return 10
        return [None for _ in chunks]

    @property
    def embedding_dimensions(self) -> int:
        if False:
            return 10
        return None