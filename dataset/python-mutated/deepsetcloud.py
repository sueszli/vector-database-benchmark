from functools import wraps
from typing import List, Optional, Union, Dict, Generator, Any
import json
import logging
import numpy as np
from haystack.document_stores import KeywordDocumentStore
from haystack.errors import HaystackError
from haystack.schema import Document, FilterType, Label
from haystack.utils import DeepsetCloud, DeepsetCloudError, args_to_kwargs
logger = logging.getLogger(__name__)

def disable_and_log(func):
    if False:
        while True:
            i = 10
    '\n    Decorator to disable write operation, shows warning and inputs instead.\n    '

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.disabled_write_warning_shown:
            logger.warning('Note that DeepsetCloudDocumentStore does not support write operations. In order to verify your pipeline works correctly, each input to write operations will be logged.')
            self.disabled_write_warning_shown = True
        args_as_kwargs = args_to_kwargs(args, func)
        parameters = {**args_as_kwargs, **kwargs}
        logger.info('Input to %s: %s', func.__name__, parameters)
    return wrapper

class DeepsetCloudDocumentStore(KeywordDocumentStore):

    def __init__(self, api_key: Optional[str]=None, workspace: str='default', index: Optional[str]=None, duplicate_documents: str='overwrite', api_endpoint: Optional[str]=None, similarity: str='dot_product', return_embedding: bool=False, label_index: str='default', embedding_dim: int=768, use_prefiltering: bool=False, search_fields: Union[str, list]='content'):
        if False:
            return 10
        '\n        A DocumentStore facade enabling you to interact with the documents stored in deepset Cloud.\n        Thus you can run experiments like trying new nodes, pipelines, etc. without having to index your data again.\n\n        You can also use this DocumentStore to create new pipelines on deepset Cloud. To do that, take the following\n        steps:\n\n        - create a new DeepsetCloudDocumentStore without an index (e.g. `DeepsetCloudDocumentStore()`)\n        - create query and indexing pipelines using this DocumentStore\n        - call `Pipeline.save_to_deepset_cloud()` passing the pipelines and a `pipeline_config_name`\n        - call `Pipeline.deploy_on_deepset_cloud()` passing the `pipeline_config_name`\n\n        DeepsetCloudDocumentStore is not intended for use in production-like scenarios.\n        See [https://haystack.deepset.ai/components/document-store](https://haystack.deepset.ai/components/document-store)\n        for more information.\n\n        :param api_key: Secret value of the API key.\n                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.\n                        See docs on how to generate an API key for your workspace: https://docs.cloud.deepset.ai/docs/connect-deepset-cloud-to-your-application\n        :param workspace: workspace name in deepset Cloud\n        :param index: name of the index to access within the deepset Cloud workspace. This equals typically the name of\n                      your pipeline. You can run Pipeline.list_pipelines_on_deepset_cloud() to see all available ones.\n                      If you set index to `None`, this DocumentStore will always return empty results.\n                      This is especially useful if you want to create a new Pipeline within deepset Cloud\n                      (see Pipeline.save_to_deepset_cloud()` and `Pipeline.deploy_on_deepset_cloud()`).\n        :param duplicate_documents: Handle duplicates document based on parameter options.\n                                    Parameter options : ( \'skip\',\'overwrite\',\'fail\')\n                                    skip: Ignore the duplicates documents\n                                    overwrite: Update any existing documents with the same ID when adding documents.\n                                    fail: an error is raised if the document ID of the document being added already\n                                    exists.\n        :param api_endpoint: The URL of the deepset Cloud API.\n                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.\n                             If DEEPSET_CLOUD_API_ENDPOINT environment variable is not specified either, defaults to "https://api.cloud.deepset.ai/api/v1".\n        :param similarity: The similarity function used to compare document vectors. \'dot_product\' is the default since it is\n                           more performant with DPR embeddings. \'cosine\' is recommended if you are using a Sentence Transformer model.\n        :param label_index: index for the evaluation set interface\n        :param return_embedding: To return document embedding.\n        :param embedding_dim: Specifies the dimensionality of the embedding vector (only needed when using a dense retriever, for example, DensePassageRetriever pr EmbeddingRetriever, on top).\n        :param use_prefiltering: By default, DeepsetCloudDocumentStore uses post-filtering when querying with filters.\n                                 To use pre-filtering instead, set this parameter to `True`. Note that pre-filtering\n                                 comes at the cost of higher latency.\n        :param search_fields: Names of fields BM25Retriever uses to find matches to the incoming query in the documents, for example: ["content", "title"].\n        '
        self.index = index
        self.label_index = label_index
        self.duplicate_documents = duplicate_documents
        self.similarity = similarity
        self.return_embedding = return_embedding
        self.embedding_dim = embedding_dim
        self.use_prefiltering = use_prefiltering
        self.search_fields = search_fields
        self.client = DeepsetCloud.get_index_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, index=index)
        pipeline_client = DeepsetCloud.get_pipeline_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        deployed_pipelines = set()
        deployed_unhealthy_pipelines = set()
        try:
            for pipe in pipeline_client.list_pipeline_configs(workspace=workspace):
                if pipe['status'] == 'DEPLOYED':
                    deployed_pipelines.add(pipe['name'])
                elif pipe['status'] == 'DEPLOYED_UNHEALTHY':
                    deployed_unhealthy_pipelines.add(pipe['name'])
        except Exception as ie:
            raise DeepsetCloudError(f'Could not connect to deepset Cloud:\n{ie}') from ie
        self.index_exists = index in deployed_pipelines | deployed_unhealthy_pipelines
        if self.index_exists:
            index_info = self.client.info()
            indexing_info = index_info['indexing']
            if indexing_info['pending_file_count'] > 0:
                logger.warning('%s files are pending to be indexed. Indexing status: %s', indexing_info['pending_file_count'], indexing_info['status'])
            if index in deployed_unhealthy_pipelines:
                logger.warning("The index '%s' is unhealthy and should be redeployed using `Pipeline.undeploy_on_deepset_cloud()` and `Pipeline.deploy_on_deepset_cloud()`.", index)
        else:
            logger.info('You are using a DeepsetCloudDocumentStore with an index that does not exist on deepset Cloud. This document store always returns empty responses. This can be useful if you want to create a new pipeline within deepset Cloud.\nIn order to create a new pipeline on deepset Cloud, take the following steps: \n  - create query and indexing pipelines using this DocumentStore\n  - call `Pipeline.save_to_deepset_cloud()` passing the pipelines and a `pipeline_config_name`\n  - call `Pipeline.deploy_on_deepset_cloud()` passing the `pipeline_config_name`')
        self.evaluation_set_client = DeepsetCloud.get_evaluation_set_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, evaluation_set=label_index)
        self.disabled_write_warning_shown = False
        super().__init__()

    def get_all_documents(self, index: Optional[str]=None, filters: Optional[FilterType]=None, return_embedding: Optional[bool]=None, batch_size: int=10000, headers: Optional[Dict[str, str]]=None) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get documents from the document store.\n\n        :param index: Name of the index to get the documents from. If None, the\n                      DocumentStore\'s default index (self.index) will be used.\n        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain\n                        conditions.\n                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical\n                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,\n                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.\n                        Logical operator keys take a dictionary of metadata field names and/or logical operators as\n                        value. Metadata field names take a dictionary of comparison operators as value. Comparison\n                        operator keys take a single value or (in case of `"$in"`) a list of values as value.\n                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison\n                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default\n                        operation.\n\n                            __Example__:\n\n                            ```python\n                            filters = {\n                                "$and": {\n                                    "type": {"$eq": "article"},\n                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},\n                                    "rating": {"$gte": 3},\n                                    "$or": {\n                                        "genre": {"$in": ["economy", "politics"]},\n                                        "publisher": {"$eq": "nytimes"}\n                                    }\n                                }\n                            }\n                            ```\n        :param return_embedding: Whether to return the document embeddings.\n        :param batch_size: Number of documents that are passed to bulk function at a time.\n        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {\'Authorization\': \'Basic YWRtaW46cm9vdA==\'} for basic authentication)\n        '
        logger.warning('`get_all_documents()` can get very slow and resource-heavy since all documents must be loaded from deepset Cloud. Consider using `get_all_documents_generator()` instead.')
        return list(self.get_all_documents_generator(index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size, headers=headers))

    def get_all_documents_generator(self, index: Optional[str]=None, filters: Optional[FilterType]=None, return_embedding: Optional[bool]=None, batch_size: int=10000, headers: Optional[Dict[str, str]]=None) -> Generator[Document, None, None]:
        if False:
            return 10
        '\n        Get documents from the document store. Under-the-hood, documents are fetched in batches from the\n        document store and yielded as individual documents. This method can be used to iteratively process\n        a large number of documents without having to load all documents in memory.\n\n        :param index: Name of the index to get the documents from. If None, the\n                      DocumentStore\'s default index (self.index) will be used.\n        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain\n                        conditions.\n                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical\n                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,\n                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.\n                        Logical operator keys take a dictionary of metadata field names and/or logical operators as\n                        value. Metadata field names take a dictionary of comparison operators as value. Comparison\n                        operator keys take a single value or (in case of `"$in"`) a list of values as value.\n                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison\n                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default\n                        operation.\n\n                            __Example__:\n\n                            ```python\n                            filters = {\n                                "$and": {\n                                    "type": {"$eq": "article"},\n                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},\n                                    "rating": {"$gte": 3},\n                                    "$or": {\n                                        "genre": {"$in": ["economy", "politics"]},\n                                        "publisher": {"$eq": "nytimes"}\n                                    }\n                                }\n                            }\n                            ```\n        :param return_embedding: Whether to return the document embeddings.\n        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.\n        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {\'Authorization\': \'Basic YWRtaW46cm9vdA==\'} for basic authentication)\n        '
        if not self.index_exists:
            return
        if batch_size != 10000:
            raise ValueError('DeepsetCloudDocumentStore does not support batching')
        if index is None:
            index = self.index
        if return_embedding is None:
            return_embedding = self.return_embedding
        raw_documents = self.client.stream_documents(return_embedding=return_embedding, filters=filters, index=index, headers=headers)
        for raw_doc in raw_documents:
            dict_doc = json.loads(raw_doc.decode('utf-8'))
            yield Document.from_dict(dict_doc)

    def get_document_by_id(self, id: str, index: Optional[str]=None, headers: Optional[Dict[str, str]]=None) -> Optional[Document]:
        if False:
            print('Hello World!')
        if not self.index_exists:
            return None
        if index is None:
            index = self.index
        doc_dict = self.client.get_document(id=id, index=index, headers=headers)
        doc: Optional[Document] = None
        if doc_dict:
            doc = Document.from_dict(doc_dict)
        return doc

    def get_documents_by_id(self, ids: List[str], index: Optional[str]=None, batch_size: int=10000, headers: Optional[Dict[str, str]]=None) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        if not self.index_exists:
            return []
        if batch_size != 10000:
            raise ValueError('DeepsetCloudDocumentStore does not support batching')
        docs = (self.get_document_by_id(id, index=index, headers=headers) for id in ids)
        return [doc for doc in docs if doc is not None]

    def get_document_count(self, filters: Optional[FilterType]=None, index: Optional[str]=None, only_documents_without_embedding: bool=False, headers: Optional[Dict[str, str]]=None) -> int:
        if False:
            while True:
                i = 10
        if not self.index_exists:
            return 0
        count_result = self.client.count_documents(filters=filters, only_documents_without_embedding=only_documents_without_embedding, index=index, headers=headers)
        return count_result['count']

    def query_by_embedding(self, query_emb: np.ndarray, filters: Optional[FilterType]=None, top_k: int=10, index: Optional[str]=None, return_embedding: Optional[bool]=None, headers: Optional[Dict[str, str]]=None, scale_score: bool=True) -> List[Document]:
        if False:
            return 10
        '\n        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.\n\n        :param query_emb: Embedding of the query (e.g. gathered from DPR)\n        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain\n                        conditions.\n                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical\n                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,\n                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.\n                        Logical operator keys take a dictionary of metadata field names and/or logical operators as\n                        value. Metadata field names take a dictionary of comparison operators as value. Comparison\n                        operator keys take a single value or (in case of `"$in"`) a list of values as value.\n                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison\n                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default\n                        operation.\n\n                            __Example__:\n\n                            ```python\n                            filters = {\n                                "$and": {\n                                    "type": {"$eq": "article"},\n                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},\n                                    "rating": {"$gte": 3},\n                                    "$or": {\n                                        "genre": {"$in": ["economy", "politics"]},\n                                        "publisher": {"$eq": "nytimes"}\n                                    }\n                                }\n                            }\n                            # or simpler using default operators\n                            filters = {\n                                "type": "article",\n                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},\n                                "rating": {"$gte": 3},\n                                "$or": {\n                                    "genre": ["economy", "politics"],\n                                    "publisher": "nytimes"\n                                }\n                            }\n                            ```\n\n                            To use the same logical operator multiple times on the same level, logical operators take\n                            optionally a list of dictionaries as value.\n\n                            __Example__:\n\n                            ```python\n                            filters = {\n                                "$or": [\n                                    {\n                                        "$and": {\n                                            "Type": "News Paper",\n                                            "Date": {\n                                                "$lt": "2019-01-01"\n                                            }\n                                        }\n                                    },\n                                    {\n                                        "$and": {\n                                            "Type": "Blog Post",\n                                            "Date": {\n                                                "$gte": "2019-01-01"\n                                            }\n                                        }\n                                    }\n                                ]\n                            }\n                            ```\n        :param top_k: How many documents to return\n        :param index: Index name for storing the docs and metadata\n        :param return_embedding: To return document embedding\n        :param headers: Custom HTTP headers to pass to requests\n        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).\n                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.\n                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.\n        :return:\n        '
        if not self.index_exists:
            return []
        if return_embedding is None:
            return_embedding = self.return_embedding
        doc_dicts = self.client.query(query_emb=query_emb.tolist(), filters=filters, top_k=top_k, return_embedding=return_embedding, index=index, scale_score=scale_score, headers=headers, use_prefiltering=self.use_prefiltering)
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def query(self, query: Optional[str], filters: Optional[FilterType]=None, top_k: int=10, custom_query: Optional[str]=None, index: Optional[str]=None, headers: Optional[Dict[str, str]]=None, all_terms_must_match: bool=False, scale_score: bool=True) -> List[Document]:
        if False:
            print('Hello World!')
        '\n        Scan through documents in DocumentStore and return a small number documents\n        that are most relevant to the query as defined by the BM25 algorithm.\n\n        :param query: The query\n        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain\n                        conditions.\n                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical\n                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,\n                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.\n                        Logical operator keys take a dictionary of metadata field names and/or logical operators as\n                        value. Metadata field names take a dictionary of comparison operators as value. Comparison\n                        operator keys take a single value or (in case of `"$in"`) a list of values as value.\n                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison\n                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default\n                        operation.\n\n                            __Example__:\n\n                            ```python\n                            filters = {\n                                "$and": {\n                                    "type": {"$eq": "article"},\n                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},\n                                    "rating": {"$gte": 3},\n                                    "$or": {\n                                        "genre": {"$in": ["economy", "politics"]},\n                                        "publisher": {"$eq": "nytimes"}\n                                    }\n                                }\n                            }\n                            # or simpler using default operators\n                            filters = {\n                                "type": "article",\n                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},\n                                "rating": {"$gte": 3},\n                                "$or": {\n                                    "genre": ["economy", "politics"],\n                                    "publisher": "nytimes"\n                                }\n                            }\n                            ```\n\n                            To use the same logical operator multiple times on the same level, logical operators take\n                            optionally a list of dictionaries as value.\n\n                            __Example__:\n\n                            ```python\n                            filters = {\n                                "$or": [\n                                    {\n                                        "$and": {\n                                            "Type": "News Paper",\n                                            "Date": {\n                                                "$lt": "2019-01-01"\n                                            }\n                                        }\n                                    },\n                                    {\n                                        "$and": {\n                                            "Type": "Blog Post",\n                                            "Date": {\n                                                "$gte": "2019-01-01"\n                                            }\n                                        }\n                                    }\n                                ]\n                            }\n                            ```\n        :param top_k: How many documents to return per query.\n        :param custom_query: Custom query to be executed.\n        :param index: The name of the index in the DocumentStore from which to retrieve documents\n        :param headers: Custom HTTP headers to pass to requests\n        :param all_terms_must_match: Whether all terms of the query must match the document.\n                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").\n                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").\n                                     Defaults to False.\n        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).\n                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.\n                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.\n        '
        if not self.index_exists:
            return []
        doc_dicts = self.client.query(query=query, filters=filters, top_k=top_k, custom_query=custom_query, index=index, all_terms_must_match=all_terms_must_match, scale_score=scale_score, headers=headers)
        docs = [Document.from_dict(doc) for doc in doc_dicts]
        return docs

    def query_batch(self, queries: List[str], filters: Optional[Union[FilterType, List[Optional[FilterType]]]]=None, top_k: int=10, custom_query: Optional[str]=None, index: Optional[str]=None, headers: Optional[Dict[str, str]]=None, all_terms_must_match: bool=False, scale_score: bool=True) -> List[List[Document]]:
        if False:
            return 10
        documents = []
        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError('Number of filters does not match number of queries. Please provide as many filters as queries or a single filter that will be applied to each query.')
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)
        for (query, cur_filters) in zip(queries, filters):
            cur_docs = self.query(query=query, filters=cur_filters, top_k=top_k, custom_query=custom_query, index=index, headers=headers, all_terms_must_match=all_terms_must_match, scale_score=scale_score)
            documents.append(cur_docs)
        return documents

    def _create_document_field_map(self) -> Dict:
        if False:
            print('Hello World!')
        return {}

    @disable_and_log
    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str]=None, batch_size: int=10000, duplicate_documents: Optional[str]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            return 10
        '\n        Indexes documents for later queries.\n\n        :param documents: a list of Python dictionaries or a list of Haystack Document objects.\n                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.\n                          Optionally: Include meta data via {"text": "<the-actual-text>",\n                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}\n                          It can be used for filtering and is accessible in the responses of the Finder.\n        :param index: Optional name of index where the documents shall be written to.\n                      If None, the DocumentStore\'s default index (self.index) will be used.\n        :param batch_size: Number of documents that are passed to bulk function at a time.\n        :param duplicate_documents: Handle duplicates document based on parameter options.\n                                    Parameter options : ( \'skip\',\'overwrite\',\'fail\')\n                                    skip: Ignore the duplicates documents\n                                    overwrite: Update any existing documents with the same ID when adding documents.\n                                    fail: an error is raised if the document ID of the document being added already\n                                    exists.\n        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {\'Authorization\': \'Basic YWRtaW46cm9vdA==\'} for basic authentication)\n\n        :return: None\n        '
        pass

    @disable_and_log
    def update_document_meta(self, id: str, meta: Dict[str, Any], index: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the metadata dictionary of a document by specifying its string id.\n\n        :param id: The ID of the Document whose metadata is being updated.\n        :param meta: A dictionary with key-value pairs that should be added / changed for the provided Document ID.\n        :param index: Name of the index the Document is located at.\n        '
        pass

    def get_evaluation_sets(self) -> List[dict]:
        if False:
            print('Hello World!')
        '\n        Returns a list of uploaded evaluation sets to deepset cloud.\n\n        :return: list of evaluation sets as dicts\n                 These contain ("name", "evaluation_set_id", "created_at", "matched_labels", "total_labels") as fields.\n        '
        return self.evaluation_set_client.get_evaluation_sets()

    def get_all_labels(self, index: Optional[str]=None, filters: Optional[FilterType]=None, headers: Optional[Dict[str, str]]=None) -> List[Label]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a list of labels for the given index name.\n\n        :param index: Optional name of evaluation set for which labels should be searched.\n                      If None, the DocumentStore's default label_index (self.label_index) will be used.\n        :filters: Not supported.\n        :param headers: Not supported.\n\n        :return: list of Labels.\n        "
        return self.evaluation_set_client.get_labels(evaluation_set=index)

    def get_label_count(self, index: Optional[str]=None, headers: Optional[Dict[str, str]]=None) -> int:
        if False:
            while True:
                i = 10
        "\n        Counts the number of labels for the given index and returns the value.\n\n        :param index: Optional evaluation set name for which the labels should be counted.\n                      If None, the DocumentStore's default label_index (self.label_index) will be used.\n        :param headers: Not supported.\n\n        :return: number of labels for the given index\n        "
        return self.evaluation_set_client.get_labels_count(evaluation_set=index)

    @disable_and_log
    def write_labels(self, labels: Union[List[Label], List[dict]], index: Optional[str]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            while True:
                i = 10
        pass

    @disable_and_log
    def delete_all_documents(self, index: Optional[str]=None, filters: Optional[FilterType]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            while True:
                i = 10
        pass

    @disable_and_log
    def delete_documents(self, index: Optional[str]=None, ids: Optional[List[str]]=None, filters: Optional[FilterType]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            return 10
        pass

    @disable_and_log
    def delete_labels(self, index: Optional[str]=None, ids: Optional[List[str]]=None, filters: Optional[FilterType]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            i = 10
            return i + 15
        pass

    @disable_and_log
    def delete_index(self, index: str):
        if False:
            print('Hello World!')
        pass