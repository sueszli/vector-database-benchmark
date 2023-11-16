from __future__ import annotations
import uuid
from mimetypes import common_types
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.conversions import common_types
from qdrant_client.models import Distance, VectorParams
from superagi.vector_store.base import VectorStore
from superagi.vector_store.document import Document
from superagi.config.config import get_config
DictFilter = Dict[str, Union[str, int, bool, dict, list]]
MetadataFilter = Union[DictFilter, common_types.Filter]

def create_qdrant_client(api_key: Optional[str]=None, url: Optional[str]=None, port: Optional[int]=None) -> QdrantClient:
    if False:
        while True:
            i = 10
    if api_key is None:
        qdrant_host_name = get_config('QDRANT_HOST_NAME') or 'localhost'
        qdrant_port = get_config('QDRANT_PORT') or 6333
        qdrant_client = QdrantClient(host=qdrant_host_name, port=qdrant_port)
    else:
        qdrant_client = QdrantClient(api_key=api_key, url=url, port=port)
    return qdrant_client

class Qdrant(VectorStore):
    """
    Qdrant vector store.

    Attributes:
        client : The Qdrant client.
        embedding_model : The embedding model.
        collection_name : The Qdrant collection.
        text_field_payload_key : Name of the field where the corresponding text for point is stored in the collection.
        metadata_payload_key : Name of the field where the corresponding metadata for point is stored in the collection.
    """
    TEXT_FIELD_KEY = 'text'
    METADATA_KEY = 'metadata'

    def __init__(self, client: QdrantClient, embedding_model: Optional[Any]=None, collection_name: str=None, text_field_payload_key: str=TEXT_FIELD_KEY, metadata_payload_key: str=METADATA_KEY):
        if False:
            for i in range(10):
                print('nop')
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.text_field_payload_key = text_field_payload_key or self.TEXT_FIELD_KEY
        self.metadata_payload_key = metadata_payload_key or self.METADATA_KEY

    def add_texts(self, input_texts: Iterable[str], metadata_list: Optional[List[dict]]=None, id_list: Optional[Sequence[str]]=None, batch_limit: int=64) -> List[str]:
        if False:
            return 10
        '\n        Add texts to the vector store.\n\n        Args:\n            input_texts : The texts to add.\n            metadata_list : The metadatas to add.\n            id_list : The ids to add.\n            batch_limit : The batch size to add.\n\n        Returns:\n            The list of ids vectors stored in Qdrant.\n        '
        collected_ids = []
        metadata_list = metadata_list or []
        id_list = id_list or [uuid.uuid4().hex for _ in input_texts]
        num_batches = len(input_texts) // batch_limit + (len(input_texts) % batch_limit != 0)
        for i in range(num_batches):
            text_batch = input_texts[i * batch_limit:(i + 1) * batch_limit]
            metadata_batch = metadata_list[i * batch_limit:(i + 1) * batch_limit] or None
            id_batch = id_list[i * batch_limit:(i + 1) * batch_limit]
            vectors = self.__get_embeddings(text_batch)
            payloads = self.__build_payloads(text_batch, metadata_batch, self.text_field_payload_key, self.metadata_payload_key)
            self.add_embeddings_to_vector_db({'ids': id_batch, 'vectors': vectors, 'payloads': payloads})
            collected_ids.extend(id_batch)
        return collected_ids

    def get_matching_text(self, text: str=None, embedding: List[float]=None, k: int=4, metadata: Optional[dict]=None, search_params: Optional[common_types.SearchParams]=None, offset: int=0, score_threshold: Optional[float]=None, consistency: Optional[common_types.ReadConsistency]=None, **kwargs: Any) -> Dict:
        if False:
            while True:
                i = 10
        '\n        Return docs most similar to query using specified search type.\n\n        Args:\n            embedding: Embedding vector to look up documents similar to.\n            k: Number of Documents to return.\n            text : The text to search.\n            filter: Filter by metadata. (Please refer https://qdrant.tech/documentation/concepts/filtering/)\n            search_params: Additional search params\n            offset: Offset of the first result to return.\n            score_threshold: Define a minimal score threshold for the result.\n            consistency: Read consistency of the search. Defines how many replicas\n                         should be queried before returning the result.\n            **kwargs : The keyword arguments to search.\n\n        Returns:\n            The list of documents most similar to the query\n        '
        if embedding is not None and text is not None:
            raise ValueError('Only provide embedding or text')
        if text is not None:
            embedding = self.__get_embeddings(text)[0]
        if metadata is not None:
            filter_conditions = []
            for (key, value) in metadata.items():
                metadata_filter = {}
                metadata_filter['key'] = key
                metadata_filter['match'] = {'value': value}
                filter_conditions.append(metadata_filter)
            filter = models.Filter(must=filter_conditions)
        try:
            results = self.client.search(collection_name=self.collection_name, query_vector=embedding, query_filter=filter, search_params=search_params, limit=k, offset=offset, with_payload=True, with_vectors=False, score_threshold=score_threshold, consistency=consistency, **kwargs)
        except Exception as err:
            raise err
        search_res = self._get_search_res(results, text)
        documents = self.__build_documents(results)
        return {'documents': documents, 'search_res': search_res}

    def get_index_stats(self) -> dict:
        if False:
            return 10
        '\n        Returns:\n            Stats or Information about a collection\n        '
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        dimensions = collection_info.config.params.vectors.size
        vector_count = collection_info.vectors_count
        return {'dimensions': dimensions, 'vector_count': vector_count}

    def add_embeddings_to_vector_db(self, embeddings: dict) -> None:
        if False:
            while True:
                i = 10
        'Upserts embeddings to the given vector store'
        try:
            self.client.upsert(collection_name=self.collection_name, points=models.Batch(ids=embeddings['ids'], vectors=embeddings['vectors'], payloads=embeddings['payload']))
        except Exception as err:
            raise err

    def delete_embeddings_from_vector_db(self, ids: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Deletes embeddings from the given vector store'
        try:
            self.client.delete(collection_name=self.collection_name, points_selector=models.PointIdsList(points=ids))
        except Exception as err:
            raise err

    def __get_embeddings(self, texts: Iterable[str]) -> List[List[float]]:
        if False:
            print('Hello World!')
        'Return embeddings for a list of texts using the embedding model.'
        if self.embedding_model is not None:
            query_vectors = []
            for text in texts:
                query_vector = self.embedding_model.get_embedding(text)
                query_vectors.append(query_vector)
        else:
            raise ValueError('Embedding model is not set')
        return query_vectors

    def __build_payloads(self, texts: Iterable[str], metadatas: Optional[List[dict]], text_field_payload_key: str, metadata_payload_key: str) -> List[dict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds and returns a list of payloads containing text and\n        corresponding metadata for each text in the input iterable.\n        '
        payloads = []
        for (i, text) in enumerate(texts):
            if text is None:
                raise ValueError('One or more of the text entries is set to None. Ensure to eliminate these before invoking the .add_texts method on the Qdrant instance.')
            metadata = metadatas[i] if metadatas is not None else None
            payloads.append({text_field_payload_key: text, metadata_payload_key: metadata})
        return payloads

    def __build_documents(self, results: List[Dict]) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        'Return the document version corresponding to each result.'
        documents = []
        for result in results:
            documents.append(Document(text_content=result.payload.get(self.text_field_payload_key), metadata=result.payload.get(self.metadata_payload_key) or {}))
        return documents

    @classmethod
    def create_collection(cls, client: QdrantClient, collection_name: str, size: int):
        if False:
            i = 10
            return i + 15
        '\n        Create a new collection in Qdrant if it does not exist.\n        \n        Args:\n            client : The Qdrant client.\n            collection_name: The name of the collection to create.\n            size: The size for the new collection.\n        '
        if not any((collection.name == collection_name for collection in client.get_collections().collections)):
            client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))

    def _get_search_res(self, results, text):
        if False:
            return 10
        contexts = [res.payload for res in results]
        i = 0
        search_res = f'Query: {text}\n'
        for context in contexts:
            search_res += f"Chunk{i}: \n{context['text']}\n"
            i += 1
        return search_res