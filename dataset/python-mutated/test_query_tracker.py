import json
import os
import time
from typing import Optional
from unittest.mock import MagicMock, Mock, patch
import pandas as pd
import pytest
from pandasai.helpers.query_exec_tracker import QueryExecTracker
from pandasai.llm.fake import FakeLLM
from pandasai.smart_dataframe import SmartDataframe
from unittest import TestCase
from datetime import datetime, timedelta
assert_almost_equal = TestCase().assertAlmostEqual

class TestQueryExecTracker:

    @pytest.fixture
    def llm(self, output: Optional[str]=None):
        if False:
            return 10
        return FakeLLM(output=output)

    @pytest.fixture
    def sample_df(self):
        if False:
            while True:
                i = 10
        return pd.DataFrame({'country': ['United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Canada', 'Australia', 'Japan', 'China'], 'gdp': [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064], 'happiness_index': [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]})

    @pytest.fixture
    def smart_dataframe(self, llm, sample_df):
        if False:
            i = 10
            return i + 15
        return SmartDataframe(sample_df, config={'llm': llm, 'enable_cache': False})

    @pytest.fixture
    def smart_datalake(self, smart_dataframe: SmartDataframe):
        if False:
            print('Hello World!')
        return smart_dataframe.lake

    @pytest.fixture
    def tracker(self):
        if False:
            print('Hello World!')
        tracker = QueryExecTracker()
        tracker.start_new_track()
        tracker.add_query_info(conversation_id='123', instance='SmartDatalake', query='which country has the highest GDP?', output_type='json')
        return tracker

    def test_add_dataframes(self, smart_dataframe: SmartDataframe, tracker: QueryExecTracker):
        if False:
            print('Hello World!')
        tracker._dataframes = []
        tracker.add_dataframes([smart_dataframe])
        assert len(tracker._dataframes) == 1
        assert len(tracker._dataframes[0]['headers']) == 3
        assert len(tracker._dataframes[0]['rows']) == 3

    def test_add_step(self, tracker: QueryExecTracker):
        if False:
            for i in range(10):
                print('nop')
        step = {'type': 'CustomStep', 'description': 'This is a custom step.'}
        tracker._steps = []
        tracker.add_step(step)
        assert len(tracker._steps) == 1
        assert tracker._steps[0] == step

    def test_format_response_dataframe(self, tracker: QueryExecTracker, sample_df: pd.DataFrame):
        if False:
            print('Hello World!')
        response = {'type': 'dataframe', 'value': sample_df}
        formatted_response = tracker._format_response(response)
        assert formatted_response['type'] == 'dataframe'
        assert len(formatted_response['value']['headers']) == 3
        assert len(formatted_response['value']['rows']) == 10

    def test_format_response_dataframe_with_datetime_field(self, tracker: QueryExecTracker, sample_df: pd.DataFrame):
        if False:
            while True:
                i = 10
        start_date = datetime(2023, 1, 1)
        date_range = [start_date + timedelta(days=x) for x in range(len(sample_df))]
        sample_df['date'] = date_range
        response = {'type': 'dataframe', 'value': sample_df}
        formatted_response = tracker._format_response(response)
        json.dumps(formatted_response)
        assert formatted_response['type'] == 'dataframe'
        assert len(formatted_response['value']['headers']) == 4
        assert len(formatted_response['value']['rows']) == 10

    def test_format_response_other_type(self, tracker: QueryExecTracker):
        if False:
            while True:
                i = 10
        response = {'type': 'other_type', 'value': 'SomeValue'}
        formatted_response = tracker._format_response(response)
        assert formatted_response['type'] == 'other_type'
        assert formatted_response['value'] == 'SomeValue'

    def test_get_summary(self):
        if False:
            for i in range(10):
                print('nop')

        def mock_function(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return 'Mock Result'
        tracker = QueryExecTracker()
        tracker.start_new_track()
        tracker.add_query_info(conversation_id='123', instance='SmartDatalake', query='which country has the highest GDP?', output_type='json')
        summary = tracker.get_summary()
        tracker.execute_func(mock_function, tag='custom_tag')
        assert 'query_info' in summary
        assert 'dataframes' in summary
        assert 'steps' in summary
        assert 'response' in summary
        assert 'execution_time' in summary
        assert 'is_related_query' in summary['query_info']

    def test_related_query_in_summary(self):
        if False:
            while True:
                i = 10

        def mock_function(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return 'Mock Result'
        tracker = QueryExecTracker()
        tracker.set_related_query(False)
        tracker.start_new_track()
        tracker.add_query_info(conversation_id='123', instance='SmartDatalake', query='which country has the highest GDP?', output_type='json')
        summary = tracker.get_summary()
        tracker.execute_func(mock_function, tag='custom_tag')
        assert 'is_related_query' in summary['query_info']
        assert not summary['query_info']['is_related_query']

    def test_get_execution_time(self, tracker: QueryExecTracker):
        if False:
            print('Hello World!')

        def mock_function(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(1)
            return 'Mock Result'
        with patch('time.time', return_value=0):
            tracker.execute_func(mock_function, tag='cache_hit')
        execution_time = tracker.get_execution_time()
        assert_almost_equal(execution_time, 1.0, delta=0.3)

    def test_execute_func_success(self, tracker: QueryExecTracker):
        if False:
            while True:
                i = 10
        tracker._steps = []
        mock_return_value = Mock()
        mock_return_value.to_string = Mock()
        mock_return_value.to_string.return_value = 'Mock Result'
        mock_func = Mock()
        mock_func.return_value = mock_return_value
        mock_func.__name__ = '_get_prompt'
        result = tracker.execute_func(mock_func, tag='_get_prompt')
        assert result.to_string() == 'Mock Result'
        assert len(tracker._steps) == 1
        step = tracker._steps[0]
        assert step['type'] == 'Generate Prompt'
        assert step['success'] is True

    def test_execute_func_failure(self, tracker: QueryExecTracker):
        if False:
            i = 10
            return i + 15

        def mock_function(*args, **kwargs):
            if False:
                print('Hello World!')
            raise Exception('Mock Exception')
        with pytest.raises(Exception):
            tracker.execute_func(mock_function, tag='custom_tag')

    def test_execute_func_cache_hit(self, tracker: QueryExecTracker):
        if False:
            print('Hello World!')
        tracker._steps = []
        mock_func = Mock()
        mock_func.return_value = 'code'
        mock_func.__name__ = 'get'
        result = tracker.execute_func(mock_func, tag='cache_hit')
        assert result == 'code'
        assert len(tracker._steps) == 1
        step = tracker._steps[0]
        assert 'code_generated' in step
        assert step['type'] == 'Cache Hit'
        assert step['success'] is True

    def test_execute_func_generate_code(self, tracker: QueryExecTracker):
        if False:
            print('Hello World!')
        tracker._steps = []
        mock_func = Mock()
        mock_func.return_value = 'code'
        mock_func.__name__ = 'generate_code'
        result = tracker.execute_func(mock_func, tag='generate_code')
        assert result == 'code'
        assert len(tracker._steps) == 1
        step = tracker._steps[0]
        assert 'code_generated' in step
        assert step['type'] == 'Generate Code'
        assert step['success'] is True

    def test_execute_func_re_rerun_code(self, tracker: QueryExecTracker):
        if False:
            return 10
        tracker._steps = []
        mock_func = Mock()
        mock_func.return_value = 'code'
        mock_func.__name__ = '_retry_run_code'
        result = tracker.execute_func(mock_func)
        result = tracker.execute_func(mock_func)
        assert result == 'code'
        assert len(tracker._steps) == 2
        step = tracker._steps[0]
        assert 'code_generated' in step
        assert step['type'] == 'Retry Code Generation (1)'
        assert step['success'] is True
        step2 = tracker._steps[1]
        assert 'code_generated' in step2
        assert step2['type'] == 'Retry Code Generation (2)'
        assert step2['success'] is True

    def test_execute_func_execute_code_success(self, sample_df: pd.DataFrame, tracker: QueryExecTracker):
        if False:
            while True:
                i = 10
        tracker._steps = []
        mock_func = Mock()
        mock_func.return_value = {'type': 'dataframe', 'value': sample_df}
        mock_func.__name__ = 'execute_code'
        result = tracker.execute_func(mock_func)
        assert result['type'] == 'dataframe'
        assert len(tracker._steps) == 1
        step = tracker._steps[0]
        assert 'result' in step
        assert step['type'] == 'Code Execution'
        assert step['success'] is True

    def test_execute_func_execute_code_fail(self, sample_df: pd.DataFrame, tracker: QueryExecTracker):
        if False:
            for i in range(10):
                print('nop')
        tracker._steps = []
        mock_func = Mock()
        mock_func.side_effect = Exception('Mock Exception')
        mock_func.__name__ = 'execute_code'
        with pytest.raises(Exception):
            tracker.execute_func(mock_func)
        assert len(tracker._steps) == 1
        step = tracker._steps[0]
        assert step['type'] == 'Code Execution'
        assert step['success'] is False

    def test_publish_method_with_server_key(self, tracker: QueryExecTracker):
        if False:
            i = 10
            return i + 15

        def mock_get_summary():
            if False:
                print('Hello World!')
            return 'Test summary data'
        tracker._server_config = {'server_url': 'http://custom-server', 'api_key': 'custom-api-key'}
        tracker.get_summary = mock_get_summary
        mock_response = MagicMock()
        mock_response.status_code = 200
        type(mock_response).text = 'Response text'
        with patch('requests.post', return_value=mock_response) as mock_post:
            result = tracker.publish()
        mock_post.assert_called_with('http://custom-server/api/log/add', json={'json_log': 'Test summary data'}, headers={'Authorization': 'Bearer custom-api-key'})
        assert result is None

    def test_publish_method_with_no_config(self, tracker: QueryExecTracker):
        if False:
            while True:
                i = 10

        def mock_get_summary():
            if False:
                while True:
                    i = 10
            return 'Test summary data'
        tracker._server_config = None
        tracker.get_summary = mock_get_summary
        mock_response = MagicMock()
        mock_response.status_code = 200
        type(mock_response).text = 'Response text'
        with patch('requests.post', return_value=mock_response) as mock_post:
            result = tracker.publish()
        mock_post.assert_not_called()
        assert result is None

    def test_publish_method_with_os_env(self, tracker: QueryExecTracker):
        if False:
            i = 10
            return i + 15

        def mock_get_summary():
            if False:
                for i in range(10):
                    print('nop')
            return 'Test summary data'
        os.environ['LOGGING_SERVER_URL'] = 'http://test-server'
        os.environ['LOGGING_SERVER_API_KEY'] = 'test-api-key'
        tracker.get_summary = mock_get_summary
        mock_response = MagicMock()
        mock_response.status_code = 200
        type(mock_response).text = 'Response text'
        with patch('requests.post', return_value=mock_response) as mock_post:
            result = tracker.publish()
        mock_post.assert_called_with('http://test-server/api/log/add', json={'json_log': 'Test summary data'}, headers={'Authorization': 'Bearer test-api-key'})
        assert result is None

    def test_multiple_instance_of_tracker(self, tracker: QueryExecTracker):
        if False:
            print('Hello World!')
        mock_func = Mock()
        mock_func.return_value = 'code'
        mock_func.__name__ = 'generate_code'
        tracker.execute_func(mock_func, tag='generate_code')
        tracker2 = QueryExecTracker()
        tracker2.start_new_track()
        tracker2.add_query_info(conversation_id='12345', instance='SmartDatalake', query='which country has the highest GDP?', output_type='json')
        assert len(tracker._steps) == 1
        assert len(tracker2._steps) == 0
        tracker2.execute_func(mock_func, tag='generate_code')
        assert len(tracker._steps) == 1
        assert len(tracker2._steps) == 1
        mock_func2 = Mock()
        mock_func2.return_value = 'code'
        mock_func2.__name__ = '_retry_run_code'
        tracker2.execute_func(mock_func2, tag='_retry_run_code')
        assert len(tracker._steps) == 1
        assert len(tracker2._steps) == 2
        assert tracker._query_info['conversation_id'] != tracker2._query_info['conversation_id']

    def test_conversation_id_in_different_tracks(self, tracker: QueryExecTracker):
        if False:
            for i in range(10):
                print('nop')
        mock_func = Mock()
        mock_func.return_value = 'code'
        mock_func.__name__ = 'generate_code'
        tracker.execute_func(mock_func, tag='generate_code')
        summary = tracker.get_summary()
        tracker.start_new_track()
        tracker.add_query_info(conversation_id='123', instance='SmartDatalake', query="Plot the GDP's?", output_type='json')
        mock_func2 = Mock()
        mock_func2.return_value = 'code'
        mock_func2.__name__ = '_retry_run_code'
        tracker.execute_func(mock_func2, tag='_retry_run_code')
        summary2 = tracker.get_summary()
        assert summary['query_info']['conversation_id'] == summary2['query_info']['conversation_id']
        assert len(tracker._steps) == 1

    def test_reasoning_answer_in_code_section(self, tracker: QueryExecTracker):
        if False:
            i = 10
            return i + 15
        mock_func = Mock()
        mock_func.return_value = ['code', 'reason', 'answer']
        mock_func.__name__ = 'generate_code'
        tracker.execute_func(mock_func, tag='generate_code')
        summary = tracker.get_summary()
        step = summary['steps'][0]
        assert 'reasoning' in step
        assert 'answer' in step
        assert step['reasoning'] == 'reason'
        assert step['answer'] == 'answer'

    def test_reasoning_answer_in_rerun_code(self, tracker: QueryExecTracker):
        if False:
            i = 10
            return i + 15
        mock_func = Mock()
        mock_func.return_value = ['code', 'reason', 'answer']
        mock_func.__name__ = '_retry_run_code'
        tracker.execute_func(mock_func, tag='_retry_run_code')
        summary = tracker.get_summary()
        step = summary['steps'][0]
        assert 'reasoning' in step
        assert 'answer' in step
        assert step['reasoning'] == 'reason'
        assert step['answer'] == 'answer'