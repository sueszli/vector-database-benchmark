from __future__ import annotations
import json
import logging
import os
import re
import shutil
from io import StringIO
from pathlib import Path
from unittest import mock
from urllib.parse import quote
import elasticsearch
import pendulum
import pytest
from airflow.configuration import conf
from airflow.providers.elasticsearch.log.es_response import ElasticSearchResponse
from airflow.providers.elasticsearch.log.es_task_handler import VALID_ES_CONFIG_KEYS, ElasticsearchTaskHandler, get_es_kwargs_from_config, getattr_nested
from airflow.utils import timezone
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.timezone import datetime
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_dags, clear_db_runs
from .elasticmock import elasticmock
from .elasticmock.utilities import SearchFailedException
pytestmark = pytest.mark.db_test
AIRFLOW_SOURCES_ROOT_DIR = Path(__file__).parents[4].resolve()
ES_PROVIDER_YAML_FILE = AIRFLOW_SOURCES_ROOT_DIR / 'airflow' / 'providers' / 'elasticsearch' / 'provider.yaml'

def get_ti(dag_id, task_id, execution_date, create_task_instance):
    if False:
        print('Hello World!')
    ti = create_task_instance(dag_id=dag_id, task_id=task_id, execution_date=execution_date, dagrun_state=DagRunState.RUNNING, state=TaskInstanceState.RUNNING)
    ti.try_number = 1
    ti.raw = False
    return ti

class TestElasticsearchTaskHandler:
    DAG_ID = 'dag_for_testing_es_task_handler'
    TASK_ID = 'task_for_testing_es_log_handler'
    EXECUTION_DATE = datetime(2016, 1, 1)
    LOG_ID = f'{DAG_ID}-{TASK_ID}-2016-01-01T00:00:00+00:00-1'
    JSON_LOG_ID = f'{DAG_ID}-{TASK_ID}-{ElasticsearchTaskHandler._clean_date(EXECUTION_DATE)}-1'
    FILENAME_TEMPLATE = '{try_number}.log'

    @pytest.fixture()
    def ti(self, create_task_instance, create_log_template):
        if False:
            return 10
        create_log_template(self.FILENAME_TEMPLATE, '{dag_id}-{task_id}-{execution_date}-{try_number}')
        yield get_ti(dag_id=self.DAG_ID, task_id=self.TASK_ID, execution_date=self.EXECUTION_DATE, create_task_instance=create_task_instance)
        clear_db_runs()
        clear_db_dags()

    @elasticmock
    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.local_log_location = 'local/log/location'
        self.end_of_log_mark = 'end_of_log\n'
        self.write_stdout = False
        self.json_format = False
        self.json_fields = 'asctime,filename,lineno,levelname,message,exc_text'
        self.host_field = 'host'
        self.offset_field = 'offset'
        self.es_task_handler = ElasticsearchTaskHandler(base_log_folder=self.local_log_location, end_of_log_mark=self.end_of_log_mark, write_stdout=self.write_stdout, json_format=self.json_format, json_fields=self.json_fields, host_field=self.host_field, offset_field=self.offset_field)
        self.es = elasticsearch.Elasticsearch('http://localhost:9200')
        self.index_name = 'test_index'
        self.doc_type = 'log'
        self.test_message = 'some random stuff'
        self.body = {'message': self.test_message, 'log_id': self.LOG_ID, 'offset': 1}
        self.es.index(index=self.index_name, doc_type=self.doc_type, body=self.body, id=1)

    def teardown_method(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.local_log_location.split(os.path.sep)[0], ignore_errors=True)

    def test_es_response(self):
        if False:
            while True:
                i = 10
        sample_response = self.es.sample_log_response()
        es_response = ElasticSearchResponse(self.es_task_handler, sample_response)
        logs_by_host = self.es_task_handler._group_logs_by_host(es_response)

        def concat_logs(lines):
            if False:
                while True:
                    i = 10
            log_range = -1 if lines[-1].message == self.es_task_handler.end_of_log_mark else None
            return '\n'.join((self.es_task_handler._format_msg(line) for line in lines[:log_range]))
        for hosted_log in logs_by_host.values():
            message = concat_logs(hosted_log)
        assert message == 'Dependencies all met for dep_context=non-requeueable deps ti=<TaskInstance: example_bash_operator.run_after_loop owen_run_run [queued]>\nStarting attempt 1 of 1\nExecuting <Task(BashOperator): run_after_loop> on 2023-07-09 07:47:32+00:00'

    @pytest.mark.parametrize('host, expected', [('http://localhost:9200', 'http://localhost:9200'), ('https://localhost:9200', 'https://localhost:9200'), ('localhost:9200', 'http://localhost:9200'), ('someurl', 'http://someurl'), ('https://', 'ValueError')])
    def test_format_url(self, host, expected):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the format_url method of the ElasticsearchTaskHandler class.\n        '
        if expected == 'ValueError':
            with pytest.raises(ValueError):
                assert ElasticsearchTaskHandler.format_url(host) == expected
        else:
            assert ElasticsearchTaskHandler.format_url(host) == expected

    def test_client(self):
        if False:
            i = 10
            return i + 15
        assert isinstance(self.es_task_handler.client, elasticsearch.Elasticsearch)
        assert self.es_task_handler.index_patterns == '_all'

    def test_client_with_config(self):
        if False:
            while True:
                i = 10
        es_conf = dict(conf.getsection('elasticsearch_configs'))
        expected_dict = {'http_compress': False, 'verify_certs': True}
        assert es_conf == expected_dict
        ElasticsearchTaskHandler(base_log_folder=self.local_log_location, end_of_log_mark=self.end_of_log_mark, write_stdout=self.write_stdout, json_format=self.json_format, json_fields=self.json_fields, host_field=self.host_field, offset_field=self.offset_field, es_kwargs=es_conf)

    def test_client_with_patterns(self):
        if False:
            for i in range(10):
                print('nop')
        patterns = 'test_*,other_*'
        handler = ElasticsearchTaskHandler(base_log_folder=self.local_log_location, end_of_log_mark=self.end_of_log_mark, write_stdout=self.write_stdout, json_format=self.json_format, json_fields=self.json_fields, host_field=self.host_field, offset_field=self.offset_field, index_patterns=patterns)
        assert handler.index_patterns == patterns

    def test_read(self, ti):
        if False:
            i = 10
            return i + 15
        ts = pendulum.now()
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert len(logs[0]) == 1
        assert self.test_message == logs[0][0][-1]
        assert not metadatas[0]['end_of_log']
        assert '1' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) > ts

    def test_read_with_patterns(self, ti):
        if False:
            while True:
                i = 10
        ts = pendulum.now()
        with mock.patch.object(self.es_task_handler, 'index_patterns', new='test_*,other_*'):
            (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert len(logs[0]) == 1
        assert self.test_message == logs[0][0][-1]
        assert not metadatas[0]['end_of_log']
        assert '1' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) > ts

    def test_read_with_patterns_no_match(self, ti):
        if False:
            return 10
        ts = pendulum.now()
        with mock.patch.object(self.es_task_handler, 'index_patterns', new='test_other_*,test_another_*'):
            (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert [[]] == logs
        assert not metadatas[0]['end_of_log']
        assert '0' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) == ts

    def test_read_with_missing_index(self, ti):
        if False:
            for i in range(10):
                print('nop')
        ts = pendulum.now()
        with mock.patch.object(self.es_task_handler, 'index_patterns', new='nonexistent,test_*'):
            with pytest.raises(elasticsearch.exceptions.NotFoundError, match='IndexMissingException.*'):
                self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})

    @pytest.mark.parametrize('seconds', [3, 6])
    def test_read_missing_logs(self, seconds, create_task_instance):
        if False:
            i = 10
            return i + 15
        "\n        When the log actually isn't there to be found, we only want to wait for 5 seconds.\n        In this case we expect to receive a message of the form 'Log {log_id} not found in elasticsearch ...'\n        "
        ti = get_ti(self.DAG_ID, self.TASK_ID, pendulum.instance(self.EXECUTION_DATE).add(days=1), create_task_instance=create_task_instance)
        ts = pendulum.now().add(seconds=-seconds)
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts)})
        assert 1 == len(logs)
        if seconds > 5:
            assert len(logs[0]) == 1
            actual_message = logs[0][0][1]
            expected_pattern = '^\\*\\*\\* Log .* not found in Elasticsearch.*'
            assert re.match(expected_pattern, actual_message) is not None
            assert metadatas[0]['end_of_log'] is True
        else:
            assert len(logs[0]) == 0
            assert logs == [[]]
            assert metadatas[0]['end_of_log'] is False
        assert len(logs) == len(metadatas)
        assert '0' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) == ts

    def test_read_with_match_phrase_query(self, ti):
        if False:
            print('Hello World!')
        similar_log_id = f'{TestElasticsearchTaskHandler.TASK_ID}-{TestElasticsearchTaskHandler.DAG_ID}-2016-01-01T00:00:00+00:00-1'
        another_test_message = 'another message'
        another_body = {'message': another_test_message, 'log_id': similar_log_id, 'offset': 1}
        self.es.index(index=self.index_name, doc_type=self.doc_type, body=another_body, id=1)
        ts = pendulum.now()
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': '0', 'last_log_timestamp': str(ts), 'end_of_log': False, 'max_offset': 2})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert self.test_message == logs[0][0][-1]
        assert another_test_message != logs[0]
        assert not metadatas[0]['end_of_log']
        assert '1' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) > ts

    def test_read_with_none_metadata(self, ti):
        if False:
            while True:
                i = 10
        (logs, metadatas) = self.es_task_handler.read(ti, 1)
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert self.test_message == logs[0][0][-1]
        assert not metadatas[0]['end_of_log']
        assert '1' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) < pendulum.now()

    def test_read_nonexistent_log(self, ti):
        if False:
            for i in range(10):
                print('nop')
        ts = pendulum.now()
        self.es.delete(index=self.index_name, doc_type=self.doc_type, id=1)
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert [[]] == logs
        assert not metadatas[0]['end_of_log']
        assert '0' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) == ts

    def test_read_with_empty_metadata(self, ti):
        if False:
            i = 10
            return i + 15
        ts = pendulum.now()
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert self.test_message == logs[0][0][-1]
        assert not metadatas[0]['end_of_log']
        assert '1' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) > ts
        self.es.delete(index=self.index_name, doc_type=self.doc_type, id=1)
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert [[]] == logs
        assert not metadatas[0]['end_of_log']
        assert '0' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) > ts

    def test_read_timeout(self, ti):
        if False:
            return 10
        ts = pendulum.now().subtract(minutes=5)
        self.es.delete(index=self.index_name, doc_type=self.doc_type, id=1)
        offset = 1
        (logs, metadatas) = self.es_task_handler.read(task_instance=ti, try_number=1, metadata={'offset': offset, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert [[]] == logs
        assert metadatas[0]['end_of_log']
        assert str(offset) == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) == ts

    def test_read_as_download_logs(self, ti):
        if False:
            print('Hello World!')
        ts = pendulum.now()
        (logs, metadatas) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'download_logs': True, 'end_of_log': False})
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert len(logs[0]) == 1
        assert self.test_message == logs[0][0][-1]
        assert not metadatas[0]['end_of_log']
        assert metadatas[0]['download_logs']
        assert '1' == metadatas[0]['offset']
        assert timezone.parse(metadatas[0]['last_log_timestamp']) > ts

    def test_read_raises(self, ti):
        if False:
            return 10
        with mock.patch.object(self.es_task_handler.log, 'exception') as mock_exception:
            with mock.patch.object(self.es_task_handler.client, 'search') as mock_execute:
                mock_execute.side_effect = SearchFailedException('Failed to read')
                (logs, metadatas) = self.es_task_handler.read(ti, 1)
            assert mock_exception.call_count == 1
            (args, kwargs) = mock_exception.call_args
            assert 'Could not read log with log_id:' in args[0]
        assert 1 == len(logs)
        assert len(logs) == len(metadatas)
        assert [[]] == logs
        assert not metadatas[0]['end_of_log']
        assert '0' == metadatas[0]['offset']

    def test_set_context(self, ti):
        if False:
            print('Hello World!')
        self.es_task_handler.set_context(ti)
        assert self.es_task_handler.mark_end_on_close

    def test_set_context_w_json_format_and_write_stdout(self, ti):
        if False:
            i = 10
            return i + 15
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.es_task_handler.formatter = formatter
        self.es_task_handler.write_stdout = True
        self.es_task_handler.json_format = True
        self.es_task_handler.set_context(ti)

    def test_read_with_json_format(self, ti):
        if False:
            i = 10
            return i + 15
        ts = pendulum.now()
        formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s - %(exc_text)s')
        self.es_task_handler.formatter = formatter
        self.es_task_handler.json_format = True
        self.body = {'message': self.test_message, 'log_id': f'{self.DAG_ID}-{self.TASK_ID}-2016_01_01T00_00_00_000000-1', 'offset': 1, 'asctime': '2020-12-24 19:25:00,962', 'filename': 'taskinstance.py', 'lineno': 851, 'levelname': 'INFO'}
        self.es_task_handler.set_context(ti)
        self.es.index(index=self.index_name, doc_type=self.doc_type, body=self.body, id=id)
        (logs, _) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert '[2020-12-24 19:25:00,962] {taskinstance.py:851} INFO - some random stuff - ' == logs[0][0][1]

    def test_read_with_json_format_with_custom_offset_and_host_fields(self, ti):
        if False:
            while True:
                i = 10
        ts = pendulum.now()
        formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s - %(exc_text)s')
        self.es_task_handler.formatter = formatter
        self.es_task_handler.json_format = True
        self.es_task_handler.host_field = 'host.name'
        self.es_task_handler.offset_field = 'log.offset'
        self.body = {'message': self.test_message, 'log_id': f'{self.DAG_ID}-{self.TASK_ID}-2016_01_01T00_00_00_000000-1', 'log': {'offset': 1}, 'host': {'name': 'somehostname'}, 'asctime': '2020-12-24 19:25:00,962', 'filename': 'taskinstance.py', 'lineno': 851, 'levelname': 'INFO'}
        self.es_task_handler.set_context(ti)
        self.es.index(index=self.index_name, doc_type=self.doc_type, body=self.body, id=id)
        (logs, _) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert '[2020-12-24 19:25:00,962] {taskinstance.py:851} INFO - some random stuff - ' == logs[0][0][1]

    def test_read_with_custom_offset_and_host_fields(self, ti):
        if False:
            for i in range(10):
                print('nop')
        ts = pendulum.now()
        self.es.delete(index=self.index_name, doc_type=self.doc_type, id=1)
        self.es_task_handler.host_field = 'host.name'
        self.es_task_handler.offset_field = 'log.offset'
        self.body = {'message': self.test_message, 'log_id': self.LOG_ID, 'log': {'offset': 1}, 'host': {'name': 'somehostname'}}
        self.es.index(index=self.index_name, doc_type=self.doc_type, body=self.body, id=id)
        (logs, _) = self.es_task_handler.read(ti, 1, {'offset': 0, 'last_log_timestamp': str(ts), 'end_of_log': False})
        assert self.test_message == logs[0][0][1]

    def test_close(self, ti):
        if False:
            i = 10
            return i + 15
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.es_task_handler.formatter = formatter
        self.es_task_handler.set_context(ti)
        self.es_task_handler.close()
        with open(os.path.join(self.local_log_location, self.FILENAME_TEMPLATE.format(try_number=1))) as log_file:
            log_line = log_file.read().strip()
            assert log_line.endswith(self.end_of_log_mark.strip())
        assert self.es_task_handler.closed

    def test_close_no_mark_end(self, ti):
        if False:
            while True:
                i = 10
        ti.raw = True
        self.es_task_handler.set_context(ti)
        self.es_task_handler.close()
        with open(os.path.join(self.local_log_location, self.FILENAME_TEMPLATE.format(try_number=1))) as log_file:
            assert self.end_of_log_mark not in log_file.read()
        assert self.es_task_handler.closed

    def test_close_closed(self, ti):
        if False:
            while True:
                i = 10
        self.es_task_handler.closed = True
        self.es_task_handler.set_context(ti)
        self.es_task_handler.close()
        with open(os.path.join(self.local_log_location, self.FILENAME_TEMPLATE.format(try_number=1))) as log_file:
            assert 0 == len(log_file.read())

    def test_close_with_no_handler(self, ti):
        if False:
            while True:
                i = 10
        self.es_task_handler.set_context(ti)
        self.es_task_handler.handler = None
        self.es_task_handler.close()
        with open(os.path.join(self.local_log_location, self.FILENAME_TEMPLATE.format(try_number=1))) as log_file:
            assert 0 == len(log_file.read())
        assert self.es_task_handler.closed

    def test_close_with_no_stream(self, ti):
        if False:
            i = 10
            return i + 15
        self.es_task_handler.set_context(ti)
        self.es_task_handler.handler.stream = None
        self.es_task_handler.close()
        with open(os.path.join(self.local_log_location, self.FILENAME_TEMPLATE.format(try_number=1))) as log_file:
            assert self.end_of_log_mark in log_file.read()
        assert self.es_task_handler.closed
        self.es_task_handler.set_context(ti)
        self.es_task_handler.handler.stream.close()
        self.es_task_handler.close()
        with open(os.path.join(self.local_log_location, self.FILENAME_TEMPLATE.format(try_number=1))) as log_file:
            assert self.end_of_log_mark in log_file.read()
        assert self.es_task_handler.closed

    def test_render_log_id(self, ti):
        if False:
            return 10
        assert self.LOG_ID == self.es_task_handler._render_log_id(ti, 1)
        self.es_task_handler.json_format = True
        assert self.JSON_LOG_ID == self.es_task_handler._render_log_id(ti, 1)

    def test_clean_date(self):
        if False:
            while True:
                i = 10
        clean_execution_date = self.es_task_handler._clean_date(datetime(2016, 7, 8, 9, 10, 11, 12))
        assert '2016_07_08T09_10_11_000012' == clean_execution_date

    @pytest.mark.parametrize('json_format, es_frontend, expected_url', [(True, 'localhost:5601/{log_id}', 'https://localhost:5601/' + quote(JSON_LOG_ID)), (False, 'localhost:5601/{log_id}', 'https://localhost:5601/' + quote(LOG_ID)), (False, 'localhost:5601', 'https://localhost:5601'), (False, 'https://localhost:5601/path/{log_id}', 'https://localhost:5601/path/' + quote(LOG_ID)), (False, 'http://localhost:5601/path/{log_id}', 'http://localhost:5601/path/' + quote(LOG_ID)), (False, 'other://localhost:5601/path/{log_id}', 'other://localhost:5601/path/' + quote(LOG_ID))])
    def test_get_external_log_url(self, ti, json_format, es_frontend, expected_url):
        if False:
            print('Hello World!')
        es_task_handler = ElasticsearchTaskHandler(base_log_folder=self.local_log_location, end_of_log_mark=self.end_of_log_mark, write_stdout=self.write_stdout, json_format=json_format, json_fields=self.json_fields, host_field=self.host_field, offset_field=self.offset_field, frontend=es_frontend)
        url = es_task_handler.get_external_log_url(ti, ti.try_number)
        assert expected_url == url

    @pytest.mark.parametrize('frontend, expected', [('localhost:5601/{log_id}', True), (None, False)])
    def test_supports_external_link(self, frontend, expected):
        if False:
            i = 10
            return i + 15
        self.es_task_handler.frontend = frontend
        assert self.es_task_handler.supports_external_link == expected

    @mock.patch('sys.__stdout__', new_callable=StringIO)
    def test_dynamic_offset(self, stdout_mock, ti, time_machine):
        if False:
            for i in range(10):
                print('nop')
        handler = ElasticsearchTaskHandler(base_log_folder=self.local_log_location, end_of_log_mark=self.end_of_log_mark, write_stdout=True, json_format=True, json_fields=self.json_fields, host_field=self.host_field, offset_field=self.offset_field)
        handler.formatter = logging.Formatter()
        logger = logging.getLogger(__name__)
        logger.handlers = [handler]
        logger.propagate = False
        ti._log = logger
        handler.set_context(ti)
        t1 = pendulum.local(year=2017, month=1, day=1, hour=1, minute=1, second=15)
        (t2, t3) = (t1 + pendulum.duration(seconds=5), t1 + pendulum.duration(seconds=10))
        time_machine.move_to(t1, tick=False)
        ti.log.info('Test')
        time_machine.move_to(t2, tick=False)
        ti.log.info('Test2')
        time_machine.move_to(t3, tick=False)
        ti.log.info('Test3')
        (first_log, second_log, third_log) = map(json.loads, stdout_mock.getvalue().strip().splitlines())
        assert first_log['offset'] < second_log['offset'] < third_log['offset']
        assert first_log['asctime'] == t1.format('YYYY-MM-DDTHH:mm:ss.SSSZZ')
        assert second_log['asctime'] == t2.format('YYYY-MM-DDTHH:mm:ss.SSSZZ')
        assert third_log['asctime'] == t3.format('YYYY-MM-DDTHH:mm:ss.SSSZZ')

def test_safe_attrgetter():
    if False:
        print('Hello World!')

    class A:
        ...
    a = A()
    a.b = 'b'
    a.c = None
    a.x = a
    a.x.d = 'blah'
    assert getattr_nested(a, 'b', None) == 'b'
    assert getattr_nested(a, 'x.d', None) == 'blah'
    assert getattr_nested(a, 'aa', 'heya') == 'heya'
    assert getattr_nested(a, 'c', 'heya') is None
    assert getattr_nested(a, 'aa', None) is None

def test_retrieve_config_keys():
    if False:
        i = 10
        return i + 15
    '\n    Tests that the ElasticsearchTaskHandler retrieves the correct configuration keys from the config file.\n    * old_parameters are removed\n    * parameters from config are automatically added\n    * constructor parameters missing from config are also added\n    :return:\n    '
    with conf_vars({('elasticsearch_configs', 'use_ssl'): 'True', ('elasticsearch_configs', 'http_compress'): 'False', ('elasticsearch_configs', 'timeout'): '10'}):
        args_from_config = get_es_kwargs_from_config().keys()
        assert 'use_ssl' not in args_from_config
        assert 'verify_certs' in args_from_config
        assert 'timeout' in args_from_config
        assert 'http_compress' in args_from_config
        assert 'self' not in args_from_config

def test_retrieve_retry_on_timeout():
    if False:
        while True:
            i = 10
    '\n    Test if retrieve timeout is converted to retry_on_timeout.\n    '
    with conf_vars({('elasticsearch_configs', 'retry_timeout'): 'True'}):
        args_from_config = get_es_kwargs_from_config().keys()
        assert 'retry_timeout' not in args_from_config
        assert 'retry_on_timeout' in args_from_config

def test_self_not_valid_arg():
    if False:
        i = 10
        return i + 15
    '\n    Test if self is not a valid argument.\n    '
    assert 'self' not in VALID_ES_CONFIG_KEYS