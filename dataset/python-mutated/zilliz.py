import logging
from typing import Dict, List, Optional, Tuple, Union
from embedchain.config import ZillizDBConfig
from embedchain.helper.json_serializable import register_deserializable
from embedchain.vectordb.base import BaseVectorDB
try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility
except ImportError:
    raise ImportError('Zilliz requires extra dependencies. Install with `pip install --upgrade embedchain[milvus]`') from None

@register_deserializable
class ZillizVectorDB(BaseVectorDB):
    """Base class for vector database."""

    def __init__(self, config: ZillizDBConfig=None):
        if False:
            while True:
                i = 10
        'Initialize the database. Save the config and client as an attribute.\n\n        :param config: Database configuration class instance.\n        :type config: ZillizDBConfig\n        '
        if config is None:
            self.config = ZillizDBConfig()
        else:
            self.config = config
        self.client = MilvusClient(uri=self.config.uri, token=self.config.token)
        self.connection = connections.connect(uri=self.config.uri, token=self.config.token)
        super().__init__(config=self.config)

    def _initialize(self):
        if False:
            print('Hello World!')
        "\n        This method is needed because `embedder` attribute needs to be set externally before it can be initialized.\n\n        So it's can't be done in __init__ in one step.\n        "
        self._get_or_create_collection(self.config.collection_name)

    def _get_or_create_db(self):
        if False:
            print('Hello World!')
        'Get or create the database.'
        return self.client

    def _get_or_create_collection(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Get or create a named collection.\n\n        :param name: Name of the collection\n        :type name: str\n        '
        if utility.has_collection(name):
            logging.info(f'[ZillizDB]: found an existing collection {name}, make sure the auto-id is disabled.')
            self.collection = Collection(name)
        else:
            fields = [FieldSchema(name='id', dtype=DataType.VARCHAR, is_primary=True, max_length=512), FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=2048), FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=self.embedder.vector_dimension)]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.collection = Collection(name=name, schema=schema)
            index = {'index_type': 'AUTOINDEX', 'metric_type': self.config.metric_type}
            self.collection.create_index('embeddings', index)
        return self.collection

    def get(self, ids: Optional[List[str]]=None, where: Optional[Dict[str, any]]=None, limit: Optional[int]=None):
        if False:
            return 10
        '\n        Get existing doc ids present in vector database\n\n        :param ids: list of doc ids to check for existence\n        :type ids: List[str]\n        :param where: Optional. to filter data\n        :type where: Dict[str, Any]\n        :param limit: Optional. maximum number of documents\n        :type limit: Optional[int]\n        :return: Existing documents.\n        :rtype: Set[str]\n        '
        if ids is None or len(ids) == 0 or self.collection.num_entities == 0:
            return {'ids': []}
        if not self.collection.is_empty:
            filter = f'id in {ids}'
            results = self.client.query(collection_name=self.config.collection_name, filter=filter, output_fields=['id'])
            results = [res['id'] for res in results]
        return {'ids': set(results)}

    def add(self, embeddings: List[List[float]], documents: List[str], metadatas: List[object], ids: List[str], skip_embedding: bool):
        if False:
            i = 10
            return i + 15
        'Add to database'
        if not skip_embedding:
            embeddings = self.embedder.embedding_fn(documents)
        for (id, doc, metadata, embedding) in zip(ids, documents, metadatas, embeddings):
            data = {**metadata, 'id': id, 'text': doc, 'embeddings': embedding}
            self.client.insert(collection_name=self.config.collection_name, data=data)
        self.collection.load()
        self.collection.flush()
        self.client.flush(self.config.collection_name)

    def query(self, input_query: List[str], n_results: int, where: Dict[str, any], skip_embedding: bool, citations: bool=False) -> Union[List[Tuple[str, str, str]], List[str]]:
        if False:
            while True:
                i = 10
        '\n        Query contents from vector data base based on vector similarity\n\n        :param input_query: list of query string\n        :type input_query: List[str]\n        :param n_results: no of similar documents to fetch from database\n        :type n_results: int\n        :param where: to filter data\n        :type where: str\n        :raises InvalidDimensionException: Dimensions do not match.\n        :param citations: we use citations boolean param to return context along with the answer.\n        :type citations: bool, default is False.\n        :return: The content of the document that matched your query,\n        along with url of the source and doc_id (if citations flag is true)\n        :rtype: List[str], if citations=False, otherwise List[Tuple[str, str, str]]\n        '
        if self.collection.is_empty:
            return []
        if not isinstance(where, str):
            where = None
        output_fields = ['text', 'url', 'doc_id']
        if skip_embedding:
            query_vector = input_query
            query_result = self.client.search(collection_name=self.config.collection_name, data=query_vector, limit=n_results, output_fields=output_fields)
        else:
            input_query_vector = self.embedder.embedding_fn([input_query])
            query_vector = input_query_vector[0]
            query_result = self.client.search(collection_name=self.config.collection_name, data=[query_vector], limit=n_results, output_fields=output_fields)
        contexts = []
        for query in query_result:
            data = query[0]['entity']
            context = data['text']
            if citations:
                source = data['url']
                doc_id = data['doc_id']
                contexts.append(tuple((context, source, doc_id)))
            else:
                contexts.append(context)
        return contexts

    def count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Count number of documents/chunks embedded in the database.\n\n        :return: number of documents\n        :rtype: int\n        '
        return self.collection.num_entities

    def reset(self, collection_names: List[str]=None):
        if False:
            print('Hello World!')
        '\n        Resets the database. Deletes all embeddings irreversibly.\n        '
        if self.config.collection_name:
            if collection_names:
                for collection_name in collection_names:
                    if collection_name in self.client.list_collections():
                        self.client.drop_collection(collection_name=collection_name)
            else:
                self.client.drop_collection(collection_name=self.config.collection_name)
                self._get_or_create_collection(self.config.collection_name)

    def set_collection_name(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the name of the collection. A collection is an isolated space for vectors.\n\n        :param name: Name of the collection.\n        :type name: str\n        '
        if not isinstance(name, str):
            raise TypeError('Collection name must be a string')
        self.config.collection_name = name

    def delete(self, keys: Union[list, str, int]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the embeddings from DB. Zilliz only support deleting with keys.\n\n\n        :param keys: Primary keys of the table entries to delete.\n        :type keys: Union[list, str, int]\n        '
        self.client.delete(collection_name=self.config.collection_name, pks=keys)