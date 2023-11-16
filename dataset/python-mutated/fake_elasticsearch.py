from __future__ import annotations
import fnmatch
import json
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from .utilities import MissingIndexException, get_random_id, query_params

class FakeElasticsearch(Elasticsearch):
    __documents_dict = None

    def __init__(self):
        if False:
            return 10
        super().__init__('http://localhost:9200')
        self.__documents_dict = {}

    @query_params()
    def ping(self, params=None):
        if False:
            return 10
        return True

    @query_params()
    def info(self, params=None):
        if False:
            for i in range(10):
                print('nop')
        return {'status': 200, 'cluster_name': 'elasticmock', 'version': {'lucene_version': '4.10.4', 'build_hash': '00f95f4ffca6de89d68b7ccaf80d148f1f70e4d4', 'number': '1.7.5', 'build_timestamp': '2016-02-02T09:55:30Z', 'build_snapshot': False}, 'name': 'Nightwatch', 'tagline': 'You Know, for Search'}

    @query_params()
    def sample_log_response(self, headers=None, params=None):
        if False:
            for i in range(10):
                print('nop')
        return {'_shards': {'failed': 0, 'skipped': 0, 'successful': 7, 'total': 7}, 'hits': {'hits': [{'_id': 'jdeZT4kBjAZqZnexVUxk', '_index': '.ds-filebeat-8.8.2-2023.07.09-000001', '_score': 2.482621, '_source': {'@timestamp': '2023-07-13T14:13:15.140Z', 'asctime': '2023-07-09T07:47:43.907+0000', 'container': {'id': 'airflow'}, 'dag_id': 'example_bash_operator', 'ecs': {'version': '8.0.0'}, 'execution_date': '2023_07_09T07_47_32_000000', 'filename': 'taskinstance.py', 'input': {'type': 'log'}, 'levelname': 'INFO', 'lineno': 1144, 'log': {'file': {'path': "/opt/airflow/Documents/GitHub/airflow/logs/dag_id=example_bash_operator'/run_id=owen_run_run/task_id=run_after_loop/attempt=1.log"}, 'offset': 0}, 'log.offset': 1688888863907337472, 'log_id': 'example_bash_operator-run_after_loop-owen_run_run--1-1', 'message': 'Dependencies all met for dep_context=non-requeueable deps ti=<TaskInstance: example_bash_operator.run_after_loop owen_run_run [queued]>', 'task_id': 'run_after_loop', 'try_number': '1'}, '_type': '_doc'}, {'_id': 'qteZT4kBjAZqZnexVUxl', '_index': '.ds-filebeat-8.8.2-2023.07.09-000001', '_score': 2.482621, '_source': {'@timestamp': '2023-07-13T14:13:15.141Z', 'asctime': '2023-07-09T07:47:43.917+0000', 'container': {'id': 'airflow'}, 'dag_id': 'example_bash_operator', 'ecs': {'version': '8.0.0'}, 'execution_date': '2023_07_09T07_47_32_000000', 'filename': 'taskinstance.py', 'input': {'type': 'log'}, 'levelname': 'INFO', 'lineno': 1347, 'log': {'file': {'path': '/opt/airflow/Documents/GitHub/airflow/logs/dag_id=example_bash_operator/run_id=owen_run_run/task_id=run_after_loop/attempt=1.log'}, 'offset': 988}, 'log.offset': 1688888863917961216, 'log_id': 'example_bash_operator-run_after_loop-owen_run_run--1-1', 'message': 'Starting attempt 1 of 1', 'task_id': 'run_after_loop', 'try_number': '1'}, '_type': '_doc'}, {'_id': 'v9eZT4kBjAZqZnexVUx2', '_index': '.ds-filebeat-8.8.2-2023.07.09-000001', '_score': 2.482621, '_source': {'@timestamp': '2023-07-13T14:13:15.143Z', 'asctime': '2023-07-09T07:47:43.928+0000', 'container': {'id': 'airflow'}, 'dag_id': 'example_bash_operator', 'ecs': {'version': '8.0.0'}, 'execution_date': '2023_07_09T07_47_32_000000', 'filename': 'taskinstance.py', 'input': {'type': 'log'}, 'levelname': 'INFO', 'lineno': 1368, 'log': {'file': {'path': '/opt/airflow/Documents/GitHub/airflow/logs/dag_id=example_bash_operator/run_id=owen_run_run/task_id=run_after_loop/attempt=1.log'}, 'offset': 1372}, 'log.offset': 1688888863928218880, 'log_id': 'example_bash_operator-run_after_loop-owen_run_run--1-1', 'message': 'Executing <Task(BashOperator): run_after_loop> on 2023-07-09 07:47:32+00:00', 'task_id': 'run_after_loop', 'try_number': '1'}, '_type': '_doc'}], 'max_score': 2.482621, 'total': {'relation': 'eq', 'value': 36}}, 'timed_out': False, 'took': 7}

    @query_params('consistency', 'op_type', 'parent', 'refresh', 'replication', 'routing', 'timeout', 'timestamp', 'ttl', 'version', 'version_type')
    def index(self, index, doc_type, body, id=None, params=None, headers=None):
        if False:
            while True:
                i = 10
        if index not in self.__documents_dict:
            self.__documents_dict[index] = []
        if id is None:
            id = get_random_id()
        version = 1
        self.__documents_dict[index].append({'_type': doc_type, '_id': id, '_source': body, '_index': index, '_version': version, '_headers': headers})
        return {'_type': doc_type, '_id': id, 'created': True, '_version': version, '_index': index, '_headers': headers}

    @query_params('parent', 'preference', 'realtime', 'refresh', 'routing')
    def exists(self, index, doc_type, id, params=None):
        if False:
            for i in range(10):
                print('nop')
        result = False
        if index in self.__documents_dict:
            for document in self.__documents_dict[index]:
                if document.get('_id') == id and document.get('_type') == doc_type:
                    result = True
                    break
        return result

    @query_params('_source', '_source_exclude', '_source_include', 'fields', 'parent', 'preference', 'realtime', 'refresh', 'routing', 'version', 'version_type')
    def get(self, index, id, doc_type='_all', params=None):
        if False:
            i = 10
            return i + 15
        result = None
        if index in self.__documents_dict:
            result = self.find_document(doc_type, id, index, result)
        if result:
            result['found'] = True
        else:
            error_data = {'_index': index, '_type': doc_type, '_id': id, 'found': False}
            raise NotFoundError(404, json.dumps(error_data))
        return result

    def find_document(self, doc_type, id, index, result):
        if False:
            while True:
                i = 10
        for document in self.__documents_dict[index]:
            if document.get('_id') == id:
                if doc_type == '_all' or document.get('_type') == doc_type:
                    result = document
                    break
        return result

    @query_params('_source', '_source_exclude', '_source_include', 'parent', 'preference', 'realtime', 'refresh', 'routing', 'version', 'version_type')
    def get_source(self, index, doc_type, id, params=None):
        if False:
            while True:
                i = 10
        document = self.get(index=index, doc_type=doc_type, id=id, params=params)
        return document.get('_source')

    @query_params('_source', '_source_exclude', '_source_include', 'allow_no_indices', 'analyze_wildcard', 'analyzer', 'default_operator', 'df', 'expand_wildcards', 'explain', 'fielddata_fields', 'fields', 'from_', 'ignore_unavailable', 'lenient', 'lowercase_expanded_terms', 'preference', 'q', 'request_cache', 'routing', 'scroll', 'search_type', 'size', 'sort', 'stats', 'suggest_field', 'suggest_mode', 'suggest_size', 'suggest_text', 'terminate_after', 'timeout', 'track_scores', 'version')
    def count(self, index=None, doc_type=None, query=None, params=None, headers=None):
        if False:
            i = 10
            return i + 15
        searchable_indexes = self._normalize_index_to_list(index, query=query)
        searchable_doc_types = self._normalize_doc_type_to_list(doc_type)
        i = 0
        for searchable_index in searchable_indexes:
            for document in self.__documents_dict[searchable_index]:
                if not searchable_doc_types or document.get('_type') in searchable_doc_types:
                    i += 1
        result = {'count': i, '_shards': {'successful': 1, 'failed': 0, 'total': 1}}
        return result

    @query_params('_source', '_source_exclude', '_source_include', 'allow_no_indices', 'analyze_wildcard', 'analyzer', 'default_operator', 'df', 'expand_wildcards', 'explain', 'fielddata_fields', 'fields', 'from_', 'ignore_unavailable', 'lenient', 'lowercase_expanded_terms', 'preference', 'q', 'request_cache', 'routing', 'scroll', 'search_type', 'size', 'sort', 'stats', 'suggest_field', 'suggest_mode', 'suggest_size', 'suggest_text', 'terminate_after', 'timeout', 'track_scores', 'version')
    def search(self, index=None, doc_type=None, query=None, params=None, headers=None):
        if False:
            print('Hello World!')
        searchable_indexes = self._normalize_index_to_list(index, query=query)
        matches = self._find_match(index, doc_type, query=query)
        result = {'hits': {'total': len(matches), 'max_score': 1.0}, '_shards': {'successful': len(searchable_indexes), 'failed': 0, 'total': len(searchable_indexes)}, 'took': 1, 'timed_out': False}
        hits = []
        for match in matches:
            match['_score'] = 1.0
            hits.append(match)
        result['hits']['hits'] = hits
        return result

    @query_params('consistency', 'parent', 'refresh', 'replication', 'routing', 'timeout', 'version', 'version_type')
    def delete(self, index, doc_type, id, params=None, headers=None):
        if False:
            return 10
        found = False
        if index in self.__documents_dict:
            for document in self.__documents_dict[index]:
                if document.get('_type') == doc_type and document.get('_id') == id:
                    found = True
                    self.__documents_dict[index].remove(document)
                    break
        result_dict = {'found': found, '_index': index, '_type': doc_type, '_id': id, '_version': 1}
        if found:
            return result_dict
        else:
            raise NotFoundError(404, json.dumps(result_dict))

    @query_params('allow_no_indices', 'expand_wildcards', 'ignore_unavailable', 'preference', 'routing')
    def suggest(self, body, index=None):
        if False:
            return 10
        if index is not None and index not in self.__documents_dict:
            raise NotFoundError(404, f'IndexMissingException[[{index}] missing]')
        result_dict = {}
        for (key, value) in body.items():
            text = value.get('text')
            suggestion = int(text) + 1 if isinstance(text, int) else f'{text}_suggestion'
            result_dict[key] = [{'text': text, 'length': 1, 'options': [{'text': suggestion, 'freq': 1, 'score': 1.0}], 'offset': 0}]
        return result_dict

    def _find_match(self, index, doc_type, query):
        if False:
            while True:
                i = 10
        searchable_indexes = self._normalize_index_to_list(index, query=query)
        searchable_doc_types = self._normalize_doc_type_to_list(doc_type)
        must = query['bool']['must'][0]
        matches = []
        for searchable_index in searchable_indexes:
            self.find_document_in_searchable_index(matches, must, searchable_doc_types, searchable_index)
        return matches

    def find_document_in_searchable_index(self, matches, must, searchable_doc_types, searchable_index):
        if False:
            print('Hello World!')
        for document in self.__documents_dict[searchable_index]:
            if not searchable_doc_types or document.get('_type') in searchable_doc_types:
                if 'match_phrase' in must:
                    self.match_must_phrase(document, matches, must)
                else:
                    matches.append(document)

    @staticmethod
    def match_must_phrase(document, matches, must):
        if False:
            i = 10
            return i + 15
        for query_id in must['match_phrase']:
            query_val = must['match_phrase'][query_id]
            if query_id in document['_source']:
                if query_val in document['_source'][query_id]:
                    matches.append(document)

    def _validate_search_targets(self, targets, query):
        if False:
            for i in range(10):
                print('nop')
        matches = set()
        for target in targets:
            print(f'Loop over:::target = {target}')
            if target in ('_all', ''):
                matches.update(self.__documents_dict)
            elif '*' in target:
                matches.update(fnmatch.filter(self.__documents_dict, target))
            elif target not in self.__documents_dict:
                raise MissingIndexException(msg=f'IndexMissingException[[{target}] missing]', query=query)
        return matches

    def _normalize_index_to_list(self, index, query):
        if False:
            print('Hello World!')
        if index is None:
            searchable_indexes = self.__documents_dict.keys()
        elif isinstance(index, str):
            searchable_indexes = [index]
        elif isinstance(index, list):
            searchable_indexes = index
        else:
            raise ValueError("Invalid param 'index'")
        generator = (target for index in searchable_indexes for target in index.split(','))
        return list(self._validate_search_targets(generator, query=query))

    @staticmethod
    def _normalize_doc_type_to_list(doc_type):
        if False:
            while True:
                i = 10
        if doc_type is None:
            searchable_doc_types = []
        elif isinstance(doc_type, str):
            searchable_doc_types = [doc_type]
        elif isinstance(doc_type, list):
            searchable_doc_types = doc_type
        else:
            raise ValueError("Invalid param 'index'")
        return searchable_doc_types