from typing import cast, Any
from langchain.schema import Document
from qdrant_client.http.models import Filter, PointIdsList, FilterSelector
from qdrant_client.local.qdrant_local import QdrantLocal
from core.index.vector_index.qdrant import Qdrant

class QdrantVectorStore(Qdrant):

    def del_texts(self, filter: Filter):
        if False:
            i = 10
            return i + 15
        if not filter:
            raise ValueError('filter must not be empty')
        self._reload_if_needed()
        self.client.delete(collection_name=self.collection_name, points_selector=FilterSelector(filter=filter))

    def del_text(self, uuid: str) -> None:
        if False:
            i = 10
            return i + 15
        self._reload_if_needed()
        self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[uuid]))

    def text_exists(self, uuid: str) -> bool:
        if False:
            while True:
                i = 10
        self._reload_if_needed()
        response = self.client.retrieve(collection_name=self.collection_name, ids=[uuid])
        return len(response) > 0

    def delete(self):
        if False:
            return 10
        self._reload_if_needed()
        self.client.delete_collection(collection_name=self.collection_name)

    def delete_group(self):
        if False:
            while True:
                i = 10
        self._reload_if_needed()
        self.client.delete_collection(collection_name=self.collection_name)

    @classmethod
    def _document_from_scored_point(cls, scored_point: Any, content_payload_key: str, metadata_payload_key: str) -> Document:
        if False:
            print('Hello World!')
        if scored_point.payload.get('doc_id'):
            return Document(page_content=scored_point.payload.get(content_payload_key), metadata={'doc_id': scored_point.id})
        return Document(page_content=scored_point.payload.get(content_payload_key), metadata=scored_point.payload.get(metadata_payload_key) or {})

    def _reload_if_needed(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.client, QdrantLocal):
            self.client = cast(QdrantLocal, self.client)
            self.client._load()