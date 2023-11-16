import copy
from typing import Union, List, Optional, Dict, Generator
import json
import logging
import warnings
from pathlib import Path
from copy import deepcopy
from inspect import Signature, signature
import numpy as np
from tqdm import tqdm
from haystack.schema import Document, FilterType
from haystack.utils.batching import get_batches_from_generator
from haystack.nodes.retriever import DenseRetriever
from haystack.document_stores.sql import SQLDocumentStore
from haystack.lazy_imports import LazyImport
with LazyImport("Run 'pip install farm-haystack[faiss]'") as faiss_import:
    import faiss
logger = logging.getLogger(__name__)

class FAISSDocumentStore(SQLDocumentStore):
    """
    A DocumentStore for very large-scale, embedding-based dense Retrievers, like the DPR.

    It implements the [FAISS library](https://github.com/facebookresearch/faiss)
    to perform similarity search on vectors.

    The document text and meta-data (for filtering) are stored using the SQLDocumentStore, while
    the vector embeddings are indexed in a FAISS index.

    When you initialize the FAISSDocumentStore, the `faiss_document_store.db` database file is created on your disk. For more information, see [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store).
    """

    def __init__(self, sql_url: str='sqlite:///faiss_document_store.db', vector_dim: Optional[int]=None, embedding_dim: int=768, faiss_index_factory_str: str='Flat', faiss_index: Optional['faiss.swigfaiss.Index']=None, return_embedding: bool=False, index: str='document', similarity: str='dot_product', embedding_field: str='embedding', progress_bar: bool=True, duplicate_documents: str='overwrite', faiss_index_path: Optional[Union[str, Path]]=None, faiss_config_path: Optional[Union[str, Path]]=None, isolation_level: Optional[str]=None, n_links: int=64, ef_search: int=20, ef_construction: int=80, validate_index_sync: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param sql_url: SQL connection URL for the database. The default value is "sqlite:///faiss_document_store.db"`. It defaults to a local, file-based SQLite DB. For large scale deployment, we recommend Postgres.\n        :param vector_dim: Deprecated. Use embedding_dim instead.\n        :param embedding_dim: The embedding vector size. Default: 768.\n        :param faiss_index_factory_str: Creates a new FAISS index of the specified type.\n                                        It determines the type based on the string you pass to it, following the conventions\n                                        of the original FAISS index factory.\n                                        Recommended options:\n                                        - "Flat" (default): Best accuracy (= exact). Becomes slow and RAM-intense for > 1 Mio docs.\n                                        - "HNSW": Graph-based heuristic. If you don\'t specify it further,\n                                                  we use the following configuration:\n                                                  HNSW64, efConstruction=80 and efSearch=20.\n                                        - "IVFx,Flat": Inverted index. Replace x with the number of centroids aka nlist.\n                                                          Rule of thumb: nlist = 10 * sqrt (num_docs) is a good starting point.\n                                        For more details see:\n                                        - [Overview of indices](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)\n                                        - [Guideline for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)\n                                        - [FAISS Index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)\n                                        Benchmarks: XXX\n        :param faiss_index: Loads an existing FAISS index. This can be an empty index you configured manually\n                            or an index with Documents you used in Haystack before and want to load again. You can use it to load a previously saved DocumentStore.\n        :param return_embedding: Returns document embedding. Unlike other document stores, FAISS will return normalized embeddings.\n        :param index: Specifies the name of the index in DocumentStore to use.\n        :param similarity: Specifies the similarity function used to compare document vectors. \'dot_product\' is the default because it\'s\n                   more performant with DPR embeddings. \'cosine\' is recommended if you\'re using a Sentence-Transformer model.\n                   In both cases, the returned values in Document.score are normalized to be in range [0,1]:\n                   For `dot_product`: expit(np.asarray(raw_score / 100))\n                   For `cosine`: (raw_score + 1) / 2\n        :param embedding_field: The name of the field containing an embedding vector.\n        :param progress_bar: Shows a tqdm progress bar.\n                             You may want to disable it in production deployments to keep the logs clean.\n        :param duplicate_documents: Handles duplicates document based on parameter options.\n                                    Parameter options: ( \'skip\',\'overwrite\',\'fail\')\n                                    skip: Ignores the duplicate documents.\n                                    overwrite: Updates any existing documents with the same ID when adding documents.\n                                    fail: Raises an error if the document ID of the document being added already\n                                    exists.\n        :param faiss_index_path: The stored FAISS index file. Call `save()` to create this file. Use the same index file path you specified when calling `save()`.\n            If you specify `faiss_index_path`, you can only pass `faiss_config_path`.\n        :param faiss_config_path: Stored FAISS initial configuration. It contains all the parameters used to initialize the DocumentStore. Call `save()` to create it and then use the same configuration file path you specified when calling `save()`. Don\'t set it if you haven\'t specified `config_path` when calling `save()`.\n        :param isolation_level: See SQLAlchemy\'s `isolation_level` parameter for [`create_engine()`](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level).\n        :param n_links: Used only if `index_factory == "HNSW"`.\n        :param ef_search: Used only if `index_factory == "HNSW"`.\n        :param ef_construction: Used only if `index_factory == "HNSW"`.\n        :param validate_index_sync: Checks if the document count equals the embedding count at initialization time.\n        '
        faiss_import.check()
        if faiss_index_path is not None:
            sig = signature(self.__class__.__init__)
            self._validate_params_load_from_disk(sig, locals())
            init_params = self._load_init_params_from_config(faiss_index_path, faiss_config_path)
            self.__class__.__init__(self, **init_params)
            return
        if similarity in ('dot_product', 'cosine'):
            self.similarity = similarity
            self.metric_type = faiss.METRIC_INNER_PRODUCT
        elif similarity == 'l2':
            self.similarity = similarity
            self.metric_type = faiss.METRIC_L2
        else:
            raise ValueError('The FAISS document store can currently only support dot_product, cosine, and l2 similarity. Set similarity to one of these values.')
        if vector_dim is not None:
            warnings.warn(message="Use `embedding_dim` as the 'vector_dim' parameter is deprecated.", category=DeprecationWarning, stacklevel=2)
            self.embedding_dim = vector_dim
        else:
            self.embedding_dim = embedding_dim
        self.faiss_index_factory_str = faiss_index_factory_str
        self.faiss_indexes: Dict[str, faiss.swigfaiss.Index] = {}
        if faiss_index:
            self.faiss_indexes[index] = faiss_index
        else:
            self.faiss_indexes[index] = self._create_new_index(embedding_dim=self.embedding_dim, index_factory=faiss_index_factory_str, metric_type=self.metric_type, n_links=n_links, ef_search=ef_search, ef_construction=ef_construction)
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        super().__init__(url=sql_url, index=index, duplicate_documents=duplicate_documents, isolation_level=isolation_level)
        if validate_index_sync:
            self._validate_index_sync()

    def _validate_params_load_from_disk(self, sig: Signature, locals: dict):
        if False:
            return 10
        allowed_params = ['faiss_index_path', 'faiss_config_path', 'self']
        invalid_param_set = False
        for param in sig.parameters.values():
            if param.name not in allowed_params and param.default != locals[param.name]:
                invalid_param_set = True
                break
        if invalid_param_set:
            raise ValueError('If faiss_index_path is passed, no other params besides faiss_config_path are allowed.')

    def _validate_index_sync(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.get_document_count() == self.get_embedding_count():
            raise ValueError(f"The number of documents in the SQL database ({self.get_document_count()}) doesn't match the number of embeddings in FAISS ({self.get_embedding_count()}). Make sure your FAISS configuration file points to the same database that you used when you saved the original index.")

    def _create_new_index(self, embedding_dim: int, metric_type, index_factory: str='Flat', n_links: int=64, ef_search: int=20, ef_construction: int=80):
        if False:
            i = 10
            return i + 15
        if index_factory == 'HNSW':
            index = faiss.IndexHNSWFlat(embedding_dim, n_links, metric_type)
            index.hnsw.efSearch = ef_search
            index.hnsw.efConstruction = ef_construction
            logger.info('HNSW params: n_links: %s, efSearch: %s, efConstruction: %s', n_links, index.hnsw.efSearch, index.hnsw.efConstruction)
        else:
            index = faiss.index_factory(embedding_dim, index_factory, metric_type)
        return index

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str]=None, batch_size: int=10000, duplicate_documents: Optional[str]=None, headers: Optional[Dict[str, str]]=None) -> None:
        if False:
            while True:
                i = 10
        "\n        Add new documents to the DocumentStore.\n\n        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index\n                          them right away in FAISS. If not, you can later call update_embeddings() to create & index them.\n        :param index: (SQL) index name for storing the docs and metadata.\n        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.\n        :param duplicate_documents: Handle duplicates document based on parameter options.\n                                    Parameter options: ( 'skip','overwrite','fail')\n                                    skip: Ignore the duplicates documents.\n                                    overwrite: Update any existing documents with the same ID when adding documents.\n                                    fail: an error is raised if the document ID of the document being added already\n                                    exists.\n        :raises DuplicateDocumentError: Exception trigger on duplicate document.\n        :return: None\n        "
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"
        if not self.faiss_indexes.get(index):
            self.faiss_indexes[index] = self._create_new_index(embedding_dim=self.embedding_dim, index_factory=self.faiss_index_factory_str, metric_type=faiss.METRIC_INNER_PRODUCT)
        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(documents=document_objects, index=index, duplicate_documents=duplicate_documents)
        if len(document_objects) == 0:
            return
        vector_id = self.faiss_indexes[index].ntotal
        add_vectors = all((doc.embedding is not None for doc in document_objects))
        if vector_id > 0 and self.duplicate_documents == 'overwrite' and add_vectors:
            logger.warning('`FAISSDocumentStore` is adding new vectors to an existing `faiss_index`.\nPlease call `update_embeddings` method to correctly repopulate `faiss_index`')
        with tqdm(total=len(document_objects), disable=not self.progress_bar, position=0, desc='Writing Documents') as progress_bar:
            for i in range(0, len(document_objects), batch_size):
                batch_documents = document_objects[i:i + batch_size]
                if add_vectors:
                    if not self.faiss_indexes[index].is_trained:
                        raise ValueError(f'FAISS index of type {self.faiss_index_factory_str} must be trained before adding vectors. Call `train_index()` method before adding the vectors. For details, refer to the documentation: [FAISSDocumentStore API](https://docs.haystack.deepset.ai/reference/document-store-api#faissdocumentstoretrain_index).')
                    embeddings = [doc.embedding for doc in batch_documents]
                    embeddings_to_index = np.array(embeddings, dtype='float32')
                    if self.similarity == 'cosine':
                        self.normalize_embedding(embeddings_to_index)
                    self.faiss_indexes[index].add(embeddings_to_index)
                elif self.duplicate_documents == 'overwrite':
                    existing_docs = self.get_documents_by_id(ids=[doc.id for doc in batch_documents], index=index)
                    existing_docs_vector_ids = {doc.id: doc.meta['vector_id'] for doc in existing_docs if doc.meta and 'vector_id' in doc.meta}
                docs_to_write_in_sql = []
                for doc in batch_documents:
                    meta = doc.meta
                    if add_vectors:
                        meta['vector_id'] = vector_id
                        vector_id += 1
                    elif self.duplicate_documents == 'overwrite' and doc.id in existing_docs_vector_ids:
                        meta['vector_id'] = existing_docs_vector_ids[doc.id]
                    docs_to_write_in_sql.append(doc)
                super(FAISSDocumentStore, self).write_documents(docs_to_write_in_sql, index=index, duplicate_documents=duplicate_documents, batch_size=batch_size)
                progress_bar.update(batch_size)

    def _create_document_field_map(self) -> Dict:
        if False:
            return 10
        return {self.index: self.embedding_field}

    def update_embeddings(self, retriever: DenseRetriever, index: Optional[str]=None, update_existing_embeddings: bool=True, filters: Optional[FilterType]=None, batch_size: int=10000):
        if False:
            i = 10
            return i + 15
        '\n        Updates the embeddings in the the document store using the encoding model specified in the retriever.\n        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).\n\n        :param retriever: Retriever to use to get embeddings for text\n        :param index: Index name for which embeddings are to be updated. If set to None, the default self.index is used.\n        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,\n                                           only documents without embeddings are processed. This mode can be used for\n                                           incremental updating of embeddings, wherein, only newly indexed documents\n                                           get processed.\n        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.\n                        Example: {"name": ["some", "more"], "category": ["only_one"]}\n        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.\n        :return: None\n        '
        index = index or self.index
        if update_existing_embeddings is True:
            if filters is None:
                self.faiss_indexes[index].reset()
                self.reset_vector_ids(index)
            else:
                raise Exception('update_existing_embeddings=True is not supported with filters.')
        if not self.faiss_indexes.get(index):
            raise ValueError("Couldn't find a FAISS index. Try to init the FAISSDocumentStore() again ...")
        if not self.faiss_indexes[index].is_trained:
            raise ValueError('FAISS index of type {} must be trained before adding vectors. Call `train_index()` method before adding the vectors. For details, refer to the documentation: [FAISSDocumentStore API](https://docs.haystack.deepset.ai/reference/document-store-api#faissdocumentstoretrain_index).'.format(self.faiss_index_factory_str))
        document_count = self.get_document_count(index=index)
        if document_count == 0:
            logger.warning('Calling DocumentStore.update_embeddings() on an empty index')
            return
        logger.info('Updating embeddings for %s docs...', document_count)
        vector_id = self.faiss_indexes[index].ntotal
        result = self._query(index=index, vector_ids=None, batch_size=batch_size, filters=filters, only_documents_without_embedding=not update_existing_embeddings)
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(total=document_count, disable=not self.progress_bar, position=0, unit=' docs', desc='Updating Embedding') as progress_bar:
            for document_batch in batched_documents:
                embeddings = retriever.embed_documents(document_batch)
                self._validate_embeddings_shape(embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim)
                if self.similarity == 'cosine':
                    self.normalize_embedding(embeddings)
                self.faiss_indexes[index].add(embeddings.astype(np.float32))
                vector_id_map = {}
                for doc in document_batch:
                    vector_id_map[str(doc.id)] = str(vector_id)
                    vector_id += 1
                self.update_vector_ids(vector_id_map, index=index)
                progress_bar.set_description_str('Documents Processed')
                progress_bar.update(batch_size)

    def get_all_documents(self, index: Optional[str]=None, filters: Optional[FilterType]=None, return_embedding: Optional[bool]=None, batch_size: int=10000, headers: Optional[Dict[str, str]]=None) -> List[Document]:
        if False:
            i = 10
            return i + 15
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        result = self.get_all_documents_generator(index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size)
        documents = list(result)
        return documents

    def get_all_documents_generator(self, index: Optional[str]=None, filters: Optional[FilterType]=None, return_embedding: Optional[bool]=None, batch_size: int=10000, headers: Optional[Dict[str, str]]=None) -> Generator[Document, None, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get all documents from the document store. Under-the-hood, documents are fetched in batches from the\n        document store and yielded as individual documents. This method can be used to iteratively process\n        a large number of documents without having to load all documents in memory.\n\n        :param index: Name of the index to get the documents from. If None, the\n                      DocumentStore\'s default index (self.index) will be used.\n        :param filters: Optional filters to narrow down the documents to return.\n                        Example: {"name": ["some", "more"], "category": ["only_one"]}\n        :param return_embedding: Whether to return the document embeddings. Unlike other document stores, FAISS will return normalized embeddings\n        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.\n        '
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        index = index or self.index
        documents = super(FAISSDocumentStore, self).get_all_documents_generator(index=index, filters=filters, batch_size=batch_size, return_embedding=False)
        if return_embedding is None:
            return_embedding = self.return_embedding
        for doc in documents:
            if return_embedding and doc.meta and (doc.meta.get('vector_id') is not None):
                doc.embedding = self.faiss_indexes[index].reconstruct(int(doc.meta['vector_id']))
            yield doc

    def get_documents_by_id(self, ids: List[str], index: Optional[str]=None, batch_size: int=10000, headers: Optional[Dict[str, str]]=None) -> List[Document]:
        if False:
            print('Hello World!')
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        index = index or self.index
        documents = super(FAISSDocumentStore, self).get_documents_by_id(ids=ids, index=index, batch_size=batch_size)
        if self.return_embedding:
            for doc in documents:
                if doc.meta and doc.meta.get('vector_id') is not None:
                    doc.embedding = self.faiss_indexes[index].reconstruct(int(doc.meta['vector_id']))
        return documents

    def get_embedding_count(self, index: Optional[str]=None, filters: Optional[FilterType]=None) -> int:
        if False:
            return 10
        '\n        Return the count of embeddings in the document store.\n        '
        if filters:
            raise Exception('filters are not supported for get_embedding_count in FAISSDocumentStore')
        index = index or self.index
        return self.faiss_indexes[index].ntotal

    def train_index(self, documents: Optional[Union[List[dict], List[Document]]]=None, embeddings: Optional[np.ndarray]=None, index: Optional[str]=None):
        if False:
            print('Hello World!')
        '\n        Some FAISS indices (e.g. IVF) require initial "training" on a sample of vectors before you can add your final vectors.\n        The train vectors should come from the same distribution as your final ones.\n        You can pass either documents (incl. embeddings) or just the plain embeddings that the index shall be trained on.\n\n        :param documents: Documents (incl. the embeddings)\n        :param embeddings: Plain embeddings\n        :param index: Name of the index to train. If None, the DocumentStore\'s default index (self.index) will be used.\n        :return: None\n        '
        index = index or self.index
        if isinstance(embeddings, np.ndarray) and documents:
            raise ValueError('Either pass `documents` or `embeddings`. You passed both.')
        if documents:
            document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
            doc_embeddings = [doc.embedding for doc in document_objects if doc.embedding is not None]
            embeddings_for_train = np.array(doc_embeddings, dtype='float32')
            self.faiss_indexes[index].train(embeddings_for_train)
        elif isinstance(embeddings, np.ndarray):
            self.faiss_indexes[index].train(embeddings)
        else:
            logger.warning('When calling `train_index`, you must provide either Documents or embeddings. Because none of these values was provided, no training will be performed. ')

    def delete_all_documents(self, index: Optional[str]=None, filters: Optional[FilterType]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            i = 10
            return i + 15
        '\n        Delete all documents from the document store.\n        '
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        logger.warning('DEPRECATION WARNINGS:\n                1. delete_all_documents() method is deprecated, please use delete_documents method\n                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045\n                ')
        self.delete_documents(index, None, filters)

    def delete_documents(self, index: Optional[str]=None, ids: Optional[List[str]]=None, filters: Optional[FilterType]=None, headers: Optional[Dict[str, str]]=None):
        if False:
            return 10
        '\n        Delete documents from the document store. All documents are deleted if no filters are passed.\n\n        :param index: Index name to delete the documents from. If None, the\n                      DocumentStore\'s default index (self.index) will be used.\n        :param ids: Optional list of IDs to narrow down the documents to be deleted.\n        :param filters: Optional filters to narrow down the documents to be deleted.\n            Example filters: {"name": ["some", "more"], "category": ["only_one"]}.\n            If filters are provided along with a list of IDs, this method deletes the\n            intersection of the two query results (documents that match the filters and\n            have their ID in the list).\n        :return: None\n        '
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        index = index or self.index
        if index in self.faiss_indexes.keys():
            if not filters and (not ids):
                self.faiss_indexes[index].reset()
            else:
                affected_docs = self.get_all_documents(filters=filters)
                if ids:
                    affected_docs = [doc for doc in affected_docs if doc.id in ids]
                doc_ids = [doc.meta.get('vector_id') for doc in affected_docs if doc.meta and doc.meta.get('vector_id') is not None]
                self.faiss_indexes[index].remove_ids(np.array(doc_ids, dtype='int64'))
        super().delete_documents(index=index, ids=ids, filters=filters)

    def delete_index(self, index: str):
        if False:
            while True:
                i = 10
        '\n        Delete an existing index. The index including all data will be removed.\n\n        :param index: The name of the index to delete.\n        :return: None\n        '
        if index == self.index:
            logger.warning("Deletion of default index '%s' detected. If you plan to use this index again, please reinstantiate '%s' in order to avoid side-effects.", index, self.__class__.__name__)
        if index in self.faiss_indexes:
            del self.faiss_indexes[index]
            logger.info("Index '%s' deleted.", index)
        super().delete_index(index)

    def query_by_embedding(self, query_emb: np.ndarray, filters: Optional[FilterType]=None, top_k: int=10, index: Optional[str]=None, return_embedding: Optional[bool]=None, headers: Optional[Dict[str, str]]=None, scale_score: bool=True) -> List[Document]:
        if False:
            i = 10
            return i + 15
        '\n        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.\n\n        :param query_emb: Embedding of the query (e.g. gathered from DPR)\n        :param filters: Optional filters to narrow down the search space.\n                        Example: {"name": ["some", "more"], "category": ["only_one"]}\n        :param top_k: How many documents to return\n        :param index: Index name to query the document from.\n        :param return_embedding: To return document embedding. Unlike other document stores, FAISS will return normalized embeddings\n        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).\n                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.\n                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.\n        :return:\n        '
        if headers:
            raise NotImplementedError('FAISSDocumentStore does not support headers.')
        if filters:
            logger.warning('Query filters are not implemented for the FAISSDocumentStore.')
        index = index or self.index
        if not self.faiss_indexes.get(index):
            raise Exception(f"Index named '{index}' does not exists. Use 'update_embeddings()' to create an index.")
        if return_embedding is None:
            return_embedding = self.return_embedding
        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        if self.similarity == 'cosine':
            self.normalize_embedding(query_emb)
        (score_matrix, vector_id_matrix) = self.faiss_indexes[index].search(query_emb, top_k)
        vector_ids_for_query = [str(vector_id) for vector_id in vector_id_matrix[0] if vector_id != -1]
        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for (v_id, s) in zip(vector_id_matrix[0], score_matrix[0])}
        return_documents = []
        for doc in documents:
            score = scores_for_vector_ids[doc.meta['vector_id']]
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            doc.score = score
            if return_embedding is True:
                doc.embedding = self.faiss_indexes[index].reconstruct(int(doc.meta['vector_id']))
            return_document = copy.copy(doc)
            return_documents.append(return_document)
        return return_documents

    def save(self, index_path: Union[str, Path], config_path: Optional[Union[str, Path]]=None):
        if False:
            return 10
        '\n        Save FAISS Index to the specified file.\n\n        The FAISS DocumentStore contains a SQL database and a FAISS index. The database is saved to your disk when you initialize the DocumentStore. The FAISS index is not. You must explicitly save it by calling the `save()` method. You can then use the saved index to load a different DocumentStore.\n\n        Saving a FAISSDocumentStore creates two files on your disk: the index file and the configuration file. The configuration file contains all the parameters needed to initialize the DocumentStore.\n        For more information, see [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store).\n\n        :param index_path: The path where you want to save the index.\n        :param config_path: The path where you want to save the configuration file. This is the JSON file that contains all the parameters to initialize the DocumentStore.\n            It defaults to the same as the index file path, except the extension (.json).\n            This file contains all the parameters passed to FAISSDocumentStore()\n            at creation time (for example the `sql_url`, `embedding_dim`, and so on), and will be\n            used by the `load()` method to restore the index with the saved configuration.\n        :return: None\n        '
        if not config_path:
            index_path = Path(index_path)
            config_path = index_path.with_suffix('.json')
        faiss.write_index(self.faiss_indexes[self.index], str(index_path))
        config_to_save = deepcopy(self._component_config['params'])
        keys_to_remove = ['faiss_index', 'faiss_index_path']
        for key in keys_to_remove:
            if key in config_to_save.keys():
                del config_to_save[key]
        with open(config_path, 'w') as ipp:
            json.dump(config_to_save, ipp, default=str)

    def _load_init_params_from_config(self, index_path: Union[str, Path], config_path: Optional[Union[str, Path]]=None):
        if False:
            return 10
        if not config_path:
            index_path = Path(index_path)
            config_path = index_path.with_suffix('.json')
        init_params: dict = {}
        try:
            with open(config_path, 'r') as ipp:
                init_params = json.load(ipp)
        except OSError as e:
            raise ValueError(f"Can't open FAISS configuration file `{config_path}`. Make sure the file exists and the you have the correct permissions to access it.") from e
        faiss_index = faiss.read_index(str(index_path))
        init_params['faiss_index'] = faiss_index
        init_params['embedding_dim'] = faiss_index.d
        return init_params

    @classmethod
    def load(cls, index_path: Union[str, Path], config_path: Optional[Union[str, Path]]=None):
        if False:
            print('Hello World!')
        '\n        Load a saved FAISS index from a file and connect to the SQL database. `load()` is a class method, so, you need to call it on the class itself instead of the instance. For more information, see [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store).\n\n        Note: To have a correct mapping from FAISS to SQL,\n              make sure to use the same SQL DB that you used when calling `save()`.\n\n        :param index_path: The stored FAISS index file. Call `save()` to create this file. Use the same index file path you specified when calling `save()`.\n        :param config_path: Stored FAISS initial configuration parameters.\n            Call `save()` to create it.\n        '
        return cls(faiss_index_path=index_path, faiss_config_path=config_path)