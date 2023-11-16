import hashlib
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain.docstore.document import Document
from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.config import AddConfig, BaseLlmConfig, ChunkerConfig
from embedchain.config.apps.base_app_config import BaseAppConfig
from embedchain.constants import SQLITE_PATH
from embedchain.data_formatter import DataFormatter
from embedchain.embedder.base import BaseEmbedder
from embedchain.helper.json_serializable import JSONSerializable
from embedchain.llm.base import BaseLlm
from embedchain.loaders.base_loader import BaseLoader
from embedchain.models.data_type import DataType, DirectDataType, IndirectDataType, SpecialDataType
from embedchain.telemetry.posthog import AnonymousTelemetry
from embedchain.utils import detect_datatype, is_valid_json_string
from embedchain.vectordb.base import BaseVectorDB
load_dotenv()

class EmbedChain(JSONSerializable):

    def __init__(self, config: BaseAppConfig, llm: BaseLlm, db: BaseVectorDB=None, embedder: BaseEmbedder=None, system_prompt: Optional[str]=None):
        if False:
            return 10
        '\n        Initializes the EmbedChain instance, sets up a vector DB client and\n        creates a collection.\n\n        :param config: Configuration just for the app, not the db or llm or embedder.\n        :type config: BaseAppConfig\n        :param llm: Instance of the LLM you want to use.\n        :type llm: BaseLlm\n        :param db: Instance of the Database to use, defaults to None\n        :type db: BaseVectorDB, optional\n        :param embedder: instance of the embedder to use, defaults to None\n        :type embedder: BaseEmbedder, optional\n        :param system_prompt: System prompt to use in the llm query, defaults to None\n        :type system_prompt: Optional[str], optional\n        :raises ValueError: No database or embedder provided.\n        '
        self.config = config
        self.llm = llm
        if db is None and (not hasattr(self.config, 'db') or self.config.db is None):
            raise ValueError('App requires Database.')
        self.db = db or self.config.db
        if embedder is None:
            raise ValueError('App requires Embedder.')
        self.embedder = embedder
        self.db._set_embedder(self.embedder)
        self.db._initialize()
        if config.collection_name:
            self.db.set_collection_name(config.collection_name)
        if system_prompt:
            self.llm.config.system_prompt = system_prompt
        self.llm.update_history(app_id=self.config.id)
        self.user_asks = []
        self.chunker: ChunkerConfig = None
        self._telemetry_props = {'class': self.__class__.__name__}
        self.telemetry = AnonymousTelemetry(enabled=self.config.collect_metrics)
        self.connection = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.cursor.execute('\n            CREATE TABLE IF NOT EXISTS data_sources (\n                pipeline_id TEXT,\n                hash TEXT,\n                type TEXT,\n                value TEXT,\n                metadata TEXT,\n                is_uploaded INTEGER DEFAULT 0,\n                PRIMARY KEY (pipeline_id, hash)\n            )\n        ')
        self.connection.commit()
        self.telemetry.capture(event_name='init', properties=self._telemetry_props)

    @property
    def collect_metrics(self):
        if False:
            print('Hello World!')
        return self.config.collect_metrics

    @collect_metrics.setter
    def collect_metrics(self, value):
        if False:
            return 10
        if not isinstance(value, bool):
            raise ValueError(f'Boolean value expected but got {type(value)}.')
        self.config.collect_metrics = value

    @property
    def online(self):
        if False:
            print('Hello World!')
        return self.llm.online

    @online.setter
    def online(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, bool):
            raise ValueError(f'Boolean value expected but got {type(value)}.')
        self.llm.online = value

    def add(self, source: Any, data_type: Optional[DataType]=None, metadata: Optional[Dict[str, Any]]=None, config: Optional[AddConfig]=None, dry_run=False, **kwargs: Dict[str, Any]):
        if False:
            return 10
        '\n        Adds the data from the given URL to the vector db.\n        Loads the data, chunks it, create embedding for each chunk\n        and then stores the embedding to vector database.\n\n        :param source: The data to embed, can be a URL, local file or raw content, depending on the data type.\n        :type source: Any\n        :param data_type: Automatically detected, but can be forced with this argument. The type of the data to add,\n        defaults to None\n        :type data_type: Optional[DataType], optional\n        :param metadata: Metadata associated with the data source., defaults to None\n        :type metadata: Optional[Dict[str, Any]], optional\n        :param config: The `AddConfig` instance to use as configuration options., defaults to None\n        :type config: Optional[AddConfig], optional\n        :raises ValueError: Invalid data type\n        :param dry_run: Optional. A dry run displays the chunks to ensure that the loader and chunker work as intended.\n        deafaults to False\n        :return: source_hash, a md5-hash of the source, in hexadecimal representation.\n        :rtype: str\n        '
        if config is not None:
            pass
        elif self.chunker is not None:
            config = AddConfig(chunker=self.chunker)
        else:
            config = AddConfig()
        try:
            DataType(source)
            logging.warning(f'''Starting from version v0.0.40, Embedchain can automatically detect the data type. So, in the `add` method, the argument order has changed. You no longer need to specify '{source}' for the `source` argument. So the code snippet will be `.add("{data_type}", "{source}")`''')
            logging.warning('Embedchain is swapping the arguments for you. This functionality might be deprecated in the future, so please adjust your code.')
            (source, data_type) = (data_type, source)
        except ValueError:
            pass
        if data_type:
            try:
                data_type = DataType(data_type)
            except ValueError:
                raise ValueError(f"Invalid data_type: '{data_type}'.", f'Please use one of the following: {[data_type.value for data_type in DataType]}') from None
        if not data_type:
            data_type = detect_datatype(source)
        hash_object = hashlib.md5(str(source).encode('utf-8'))
        source_hash = hash_object.hexdigest()
        self.cursor.execute('SELECT 1 FROM data_sources WHERE hash = ? AND pipeline_id = ?', (source_hash, self.config.id))
        existing_data = self.cursor.fetchone()
        if existing_data:
            print(f'Data with hash {source_hash} already exists. Skipping addition.')
            return source_hash
        self.user_asks.append([source, data_type.value, metadata])
        data_formatter = DataFormatter(data_type, config, kwargs)
        (documents, metadatas, _ids, new_chunks) = self._load_and_embed(data_formatter.loader, data_formatter.chunker, source, metadata, source_hash, dry_run)
        if data_type in {DataType.DOCS_SITE}:
            self.is_docs_site_instance = True
        self.cursor.execute('\n            INSERT INTO data_sources (hash, pipeline_id, type, value, metadata)\n            VALUES (?, ?, ?, ?, ?)\n        ', (source_hash, self.config.id, data_type.value, str(source), json.dumps(metadata)))
        self.connection.commit()
        if dry_run:
            data_chunks_info = {'chunks': documents, 'metadata': metadatas, 'count': len(documents), 'type': data_type}
            logging.debug(f'Dry run info : {data_chunks_info}')
            return data_chunks_info
        if self.config.collect_metrics:
            word_count = data_formatter.chunker.get_word_count(documents)
            event_properties = {**self._telemetry_props, 'data_type': data_type.value, 'word_count': word_count, 'chunks_count': new_chunks}
            self.telemetry.capture(event_name='add', properties=event_properties)
        return source_hash

    def add_local(self, source: Any, data_type: Optional[DataType]=None, metadata: Optional[Dict[str, Any]]=None, config: Optional[AddConfig]=None, **kwargs: Dict[str, Any]):
        if False:
            return 10
        '\n        Adds the data from the given URL to the vector db.\n        Loads the data, chunks it, create embedding for each chunk\n        and then stores the embedding to vector database.\n\n        Warning:\n            This method is deprecated and will be removed in future versions. Use `add` instead.\n\n        :param source: The data to embed, can be a URL, local file or raw content, depending on the data type.\n        :type source: Any\n        :param data_type: Automatically detected, but can be forced with this argument. The type of the data to add,\n        defaults to None\n        :type data_type: Optional[DataType], optional\n        :param metadata: Metadata associated with the data source., defaults to None\n        :type metadata: Optional[Dict[str, Any]], optional\n        :param config: The `AddConfig` instance to use as configuration options., defaults to None\n        :type config: Optional[AddConfig], optional\n        :raises ValueError: Invalid data type\n        :return: source_hash, a md5-hash of the source, in hexadecimal representation.\n        :rtype: str\n        '
        logging.warning('The `add_local` method is deprecated and will be removed in future versions. Please use the `add` method for both local and remote files.')
        return self.add(source=source, data_type=data_type, metadata=metadata, config=config, kwargs=kwargs)

    def _get_existing_doc_id(self, chunker: BaseChunker, src: Any):
        if False:
            return 10
        '\n        Get id of existing document for a given source, based on the data type\n        '
        if chunker.data_type.value in [item.value for item in DirectDataType]:
            return None
        elif chunker.data_type.value in [item.value for item in IndirectDataType]:
            where = {'url': src}
            if chunker.data_type == DataType.JSON and is_valid_json_string(src):
                url = hashlib.sha256(src.encode('utf-8')).hexdigest()
                where = {'url': url}
            if self.config.id is not None:
                where.update({'app_id': self.config.id})
            existing_embeddings = self.db.get(where=where, limit=1)
            if len(existing_embeddings.get('metadatas', [])) > 0:
                return existing_embeddings['metadatas'][0]['doc_id']
            else:
                return None
        elif chunker.data_type.value in [item.value for item in SpecialDataType]:
            if chunker.data_type == DataType.QNA_PAIR:
                where = {'question': src[0]}
                if self.config.id is not None:
                    where.update({'app_id': self.config.id})
                existing_embeddings = self.db.get(where=where, limit=1)
                if len(existing_embeddings.get('metadatas', [])) > 0:
                    return existing_embeddings['metadatas'][0]['doc_id']
                else:
                    return None
            else:
                raise NotImplementedError(f'SpecialDataType {chunker.data_type} must have a custom logic to check for existing data')
        else:
            raise TypeError(f'{chunker.data_type} is type {type(chunker.data_type)}. When it should be  DirectDataType, IndirectDataType or SpecialDataType.')

    def _load_and_embed(self, loader: BaseLoader, chunker: BaseChunker, src: Any, metadata: Optional[Dict[str, Any]]=None, source_hash: Optional[str]=None, dry_run=False):
        if False:
            i = 10
            return i + 15
        "\n        Loads the data from the given URL, chunks it, and adds it to database.\n\n        :param loader: The loader to use to load the data.\n        :param chunker: The chunker to use to chunk the data.\n        :param src: The data to be handled by the loader. Can be a URL for\n        remote sources or local content for local loaders.\n        :param metadata: Optional. Metadata associated with the data source.\n        :param source_hash: Hexadecimal hash of the source.\n        :param dry_run: Optional. A dry run returns chunks and doesn't update DB.\n        :type dry_run: bool, defaults to False\n        :return: (List) documents (embedded text), (List) metadata, (list) ids, (int) number of chunks\n        "
        existing_doc_id = self._get_existing_doc_id(chunker=chunker, src=src)
        app_id = self.config.id if self.config is not None else None
        embeddings_data = chunker.create_chunks(loader, src, app_id=app_id)
        documents = embeddings_data['documents']
        metadatas = embeddings_data['metadatas']
        ids = embeddings_data['ids']
        new_doc_id = embeddings_data['doc_id']
        if existing_doc_id and existing_doc_id == new_doc_id:
            print('Doc content has not changed. Skipping creating chunks and embeddings')
            return ([], [], [], 0)
        if existing_doc_id and existing_doc_id != new_doc_id:
            print('Doc content has changed. Recomputing chunks and embeddings intelligently.')
            self.db.delete({'doc_id': existing_doc_id})
        where = {'url': src}
        if chunker.data_type == DataType.JSON and is_valid_json_string(src):
            url = hashlib.sha256(src.encode('utf-8')).hexdigest()
            where = {'url': url}
        if chunker.data_type == DataType.QNA_PAIR:
            where = {'question': src[0]}
        if self.config.id is not None:
            where['app_id'] = self.config.id
        db_result = self.db.get(ids=ids, where=where)
        existing_ids = set(db_result['ids'])
        if len(existing_ids):
            data_dict = {id: (doc, meta) for (id, doc, meta) in zip(ids, documents, metadatas)}
            data_dict = {id: value for (id, value) in data_dict.items() if id not in existing_ids}
            if not data_dict:
                src_copy = src
                if len(src_copy) > 50:
                    src_copy = src[:50] + '...'
                print(f'All data from {src_copy} already exists in the database.')
                return ([], [], [], 0)
            ids = list(data_dict.keys())
            (documents, metadatas) = zip(*data_dict.values())
        new_metadatas = []
        for m in metadatas:
            if self.config.id:
                m['app_id'] = self.config.id
            m['hash'] = source_hash
            if metadata:
                m.update(metadata)
            new_metadatas.append(m)
        metadatas = new_metadatas
        if dry_run:
            return (list(documents), metadatas, ids, 0)
        chunks_before_addition = self.db.count()
        self.db.add(embeddings=embeddings_data.get('embeddings', None), documents=documents, metadatas=metadatas, ids=ids, skip_embedding=chunker.data_type == DataType.IMAGES)
        count_new_chunks = self.db.count() - chunks_before_addition
        print(f'Successfully saved {src} ({chunker.data_type}). New chunks count: {count_new_chunks}')
        return (list(documents), metadatas, ids, count_new_chunks)

    def _format_result(self, results):
        if False:
            print('Hello World!')
        return [(Document(page_content=result[0], metadata=result[1] or {}), result[2]) for result in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])]

    def _retrieve_from_database(self, input_query: str, config: Optional[BaseLlmConfig]=None, where=None, citations: bool=False) -> Union[List[Tuple[str, str, str]], List[str]]:
        if False:
            print('Hello World!')
        '\n        Queries the vector database based on the given input query.\n        Gets relevant doc based on the query\n\n        :param input_query: The query to use.\n        :type input_query: str\n        :param config: The query configuration, defaults to None\n        :type config: Optional[BaseLlmConfig], optional\n        :param where: A dictionary of key-value pairs to filter the database results, defaults to None\n        :type where: _type_, optional\n        :param citations: A boolean to indicate if db should fetch citation source\n        :type citations: bool\n        :return: List of contents of the document that matched your query\n        :rtype: List[str]\n        '
        query_config = config or self.llm.config
        if where is not None:
            where = where
        else:
            where = {}
            if query_config is not None and query_config.where is not None:
                where = query_config.where
            if self.config.id is not None:
                where.update({'app_id': self.config.id})
        db_query = input_query
        if hasattr(config, 'query_type') and config.query_type == 'Images':
            from embedchain.models.clip_processor import ClipProcessor
            db_query = ClipProcessor.get_text_features(query=input_query)
        contexts = self.db.query(input_query=db_query, n_results=query_config.number_documents, where=where, skip_embedding=hasattr(config, 'query_type') and config.query_type == 'Images', citations=citations)
        return contexts

    def query(self, input_query: str, config: BaseLlmConfig=None, dry_run=False, where: Optional[Dict]=None, **kwargs: Dict[str, Any]) -> Union[Tuple[str, List[Tuple[str, str, str]]], str]:
        if False:
            i = 10
            return i + 15
        '\n        Queries the vector database based on the given input query.\n        Gets relevant doc based on the query and then passes it to an\n        LLM as context to get the answer.\n\n        :param input_query: The query to use.\n        :type input_query: str\n        :param config: The `BaseLlmConfig` instance to use as configuration options. This is used for one method call.\n        To persistently use a config, declare it during app init., defaults to None\n        :type config: Optional[BaseLlmConfig], optional\n        :param dry_run: A dry run does everything except send the resulting prompt to\n        the LLM. The purpose is to test the prompt, not the response., defaults to False\n        :type dry_run: bool, optional\n        :param where: A dictionary of key-value pairs to filter the database results., defaults to None\n        :type where: Optional[Dict[str, str]], optional\n        :param kwargs: To read more params for the query function. Ex. we use citations boolean\n        param to return context along with the answer\n        :type kwargs: Dict[str, Any]\n        :return: The answer to the query, with citations if the citation flag is True\n        or the dry run result\n        :rtype: str, if citations is False, otherwise Tuple[str,List[Tuple[str,str,str]]]\n        '
        citations = kwargs.get('citations', False)
        contexts = self._retrieve_from_database(input_query=input_query, config=config, where=where, citations=citations)
        if citations and len(contexts) > 0 and isinstance(contexts[0], tuple):
            contexts_data_for_llm_query = list(map(lambda x: x[0], contexts))
        else:
            contexts_data_for_llm_query = contexts
        answer = self.llm.query(input_query=input_query, contexts=contexts_data_for_llm_query, config=config, dry_run=dry_run)
        self.telemetry.capture(event_name='query', properties=self._telemetry_props)
        if citations:
            return (answer, contexts)
        else:
            return answer

    def chat(self, input_query: str, config: Optional[BaseLlmConfig]=None, dry_run=False, where: Optional[Dict[str, str]]=None, **kwargs: Dict[str, Any]) -> str:
        if False:
            while True:
                i = 10
        '\n        Queries the vector database on the given input query.\n        Gets relevant doc based on the query and then passes it to an\n        LLM as context to get the answer.\n\n        Maintains the whole conversation in memory.\n\n        :param input_query: The query to use.\n        :type input_query: str\n        :param config: The `BaseLlmConfig` instance to use as configuration options. This is used for one method call.\n        To persistently use a config, declare it during app init., defaults to None\n        :type config: Optional[BaseLlmConfig], optional\n        :param dry_run: A dry run does everything except send the resulting prompt to\n        the LLM. The purpose is to test the prompt, not the response., defaults to False\n        :type dry_run: bool, optional\n        :param where: A dictionary of key-value pairs to filter the database results., defaults to None\n        :type where: Optional[Dict[str, str]], optional\n        :param kwargs: To read more params for the query function. Ex. we use citations boolean\n        param to return context along with the answer\n        :type kwargs: Dict[str, Any]\n        :return: The answer to the query, with citations if the citation flag is True\n        or the dry run result\n        :rtype: str, if citations is False, otherwise Tuple[str,List[Tuple[str,str,str]]]\n        '
        citations = kwargs.get('citations', False)
        contexts = self._retrieve_from_database(input_query=input_query, config=config, where=where, citations=citations)
        if citations and len(contexts) > 0 and isinstance(contexts[0], tuple):
            contexts_data_for_llm_query = list(map(lambda x: x[0], contexts))
        else:
            contexts_data_for_llm_query = contexts
        answer = self.llm.chat(input_query=input_query, contexts=contexts_data_for_llm_query, config=config, dry_run=dry_run)
        self.llm.add_history(self.config.id, input_query, answer)
        self.telemetry.capture(event_name='chat', properties=self._telemetry_props)
        if citations:
            return (answer, contexts)
        else:
            return answer

    def set_collection_name(self, name: str):
        if False:
            print('Hello World!')
        '\n        Set the name of the collection. A collection is an isolated space for vectors.\n\n        Using `app.db.set_collection_name` method is preferred to this.\n\n        :param name: Name of the collection.\n        :type name: str\n        '
        self.db.set_collection_name(name)
        self.db._get_or_create_collection(name)

    def count(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Count the number of embeddings.\n\n        DEPRECATED IN FAVOR OF `db.count()`\n\n        :return: The number of embeddings.\n        :rtype: int\n        '
        logging.warning('DEPRECATION WARNING: Please use `app.db.count()` instead of `app.count()`.')
        return self.db.count()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resets the database. Deletes all embeddings irreversibly.\n        `App` does not have to be reinitialized after using this method.\n        '
        self.db.reset()
        self.cursor.execute('DELETE FROM data_sources WHERE pipeline_id = ?', (self.config.id,))
        self.connection.commit()
        self.delete_history()
        self.telemetry.capture(event_name='reset', properties=self._telemetry_props)

    def get_history(self, num_rounds: int=10, display_format: bool=True):
        if False:
            return 10
        return self.llm.memory.get_recent_memories(app_id=self.config.id, num_rounds=num_rounds, display_format=display_format)

    def delete_history(self):
        if False:
            print('Hello World!')
        self.llm.memory.delete_chat_history(app_id=self.config.id)