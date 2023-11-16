import json
import logging
import tempfile
from collections import defaultdict
from contextlib import nullcontext as does_not_raise
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import pytest
import requests
from airbyte_cdk.models import AirbyteGlobalState, AirbyteStateBlob, AirbyteStateMessage, AirbyteStateType, AirbyteStreamState, ConfiguredAirbyteCatalog, StreamDescriptor, SyncMode, Type
from airbyte_cdk.sources import AbstractSource, Source
from airbyte_cdk.sources.streams.core import Stream
from airbyte_cdk.sources.streams.http.availability_strategy import HttpAvailabilityStrategy
from airbyte_cdk.sources.streams.http.http import HttpStream, HttpSubStream
from airbyte_cdk.sources.utils.transform import TransformConfig, TypeTransformer
from pydantic import ValidationError

class MockSource(Source):

    def read(self, logger: logging.Logger, config: Mapping[str, Any], catalog: ConfiguredAirbyteCatalog, state: MutableMapping[str, Any]=None):
        if False:
            i = 10
            return i + 15
        pass

    def check(self, logger: logging.Logger, config: Mapping[str, Any]):
        if False:
            return 10
        pass

    def discover(self, logger: logging.Logger, config: Mapping[str, Any]):
        if False:
            return 10
        pass

class MockAbstractSource(AbstractSource):

    def __init__(self, streams: Optional[List[Stream]]=None):
        if False:
            i = 10
            return i + 15
        self._streams = streams

    def check_connection(self, *args, **kwargs) -> Tuple[bool, Optional[Any]]:
        if False:
            return 10
        return (True, '')

    def streams(self, *args, **kwargs) -> List[Stream]:
        if False:
            return 10
        if self._streams:
            return self._streams
        return []

@pytest.fixture
def source():
    if False:
        i = 10
        return i + 15
    return MockSource()

@pytest.fixture
def catalog():
    if False:
        return 10
    configured_catalog = {'streams': [{'stream': {'name': 'mock_http_stream', 'json_schema': {}, 'supported_sync_modes': ['full_refresh']}, 'destination_sync_mode': 'overwrite', 'sync_mode': 'full_refresh'}, {'stream': {'name': 'mock_stream', 'json_schema': {}, 'supported_sync_modes': ['full_refresh']}, 'destination_sync_mode': 'overwrite', 'sync_mode': 'full_refresh'}]}
    return ConfiguredAirbyteCatalog.parse_obj(configured_catalog)

@pytest.fixture
def abstract_source(mocker):
    if False:
        print('Hello World!')
    mocker.patch.multiple(HttpStream, __abstractmethods__=set())
    mocker.patch.multiple(Stream, __abstractmethods__=set())

    class MockHttpStream(mocker.MagicMock, HttpStream):
        url_base = 'http://example.com'
        path = '/dummy/path'
        get_json_schema = mocker.MagicMock()

        def supports_incremental(self):
            if False:
                print('Hello World!')
            return True

        def __init__(self, *args, **kvargs):
            if False:
                print('Hello World!')
            mocker.MagicMock.__init__(self)
            HttpStream.__init__(self, *args, kvargs)
            self.read_records = mocker.MagicMock()

        @property
        def availability_strategy(self):
            if False:
                for i in range(10):
                    print('nop')
            return None

    class MockStream(mocker.MagicMock, Stream):
        page_size = None
        get_json_schema = mocker.MagicMock()

        def __init__(self, **kwargs):
            if False:
                return 10
            mocker.MagicMock.__init__(self)
            self.read_records = mocker.MagicMock()
    streams = [MockHttpStream(), MockStream()]

    class MockAbstractSource(AbstractSource):

        def check_connection(self):
            if False:
                return 10
            return (True, None)

        def streams(self, config):
            if False:
                for i in range(10):
                    print('nop')
            self.streams_config = config
            return streams
    return MockAbstractSource()

@pytest.mark.parametrize('incoming_state, expected_state, expected_error', [pytest.param([{'type': 'STREAM', 'stream': {'stream_state': {'created_at': '2009-07-19'}, 'stream_descriptor': {'name': 'movies', 'namespace': 'public'}}}], [AirbyteStateMessage(type=AirbyteStateType.STREAM, stream=AirbyteStreamState(stream_descriptor=StreamDescriptor(name='movies', namespace='public'), stream_state=AirbyteStateBlob.parse_obj({'created_at': '2009-07-19'})))], does_not_raise(), id='test_incoming_stream_state'), pytest.param([{'type': 'STREAM', 'stream': {'stream_state': {'created_at': '2009-07-19'}, 'stream_descriptor': {'name': 'movies', 'namespace': 'public'}}}, {'type': 'STREAM', 'stream': {'stream_state': {'id': 'villeneuve_denis'}, 'stream_descriptor': {'name': 'directors', 'namespace': 'public'}}}, {'type': 'STREAM', 'stream': {'stream_state': {'created_at': '1995-12-27'}, 'stream_descriptor': {'name': 'actors', 'namespace': 'public'}}}], [AirbyteStateMessage(type=AirbyteStateType.STREAM, stream=AirbyteStreamState(stream_descriptor=StreamDescriptor(name='movies', namespace='public'), stream_state=AirbyteStateBlob.parse_obj({'created_at': '2009-07-19'}))), AirbyteStateMessage(type=AirbyteStateType.STREAM, stream=AirbyteStreamState(stream_descriptor=StreamDescriptor(name='directors', namespace='public'), stream_state=AirbyteStateBlob.parse_obj({'id': 'villeneuve_denis'}))), AirbyteStateMessage(type=AirbyteStateType.STREAM, stream=AirbyteStreamState(stream_descriptor=StreamDescriptor(name='actors', namespace='public'), stream_state=AirbyteStateBlob.parse_obj({'created_at': '1995-12-27'})))], does_not_raise(), id='test_incoming_multiple_stream_states'), pytest.param([{'type': 'GLOBAL', 'global': {'shared_state': {'shared_key': 'shared_val'}, 'stream_states': [{'stream_state': {'created_at': '2009-07-19'}, 'stream_descriptor': {'name': 'movies', 'namespace': 'public'}}]}}], [AirbyteStateMessage.parse_obj({'type': AirbyteStateType.GLOBAL, 'global': AirbyteGlobalState(shared_state=AirbyteStateBlob.parse_obj({'shared_key': 'shared_val'}), stream_states=[AirbyteStreamState(stream_descriptor=StreamDescriptor(name='movies', namespace='public'), stream_state=AirbyteStateBlob.parse_obj({'created_at': '2009-07-19'}))])})], does_not_raise(), id='test_incoming_global_state'), pytest.param({'movies': {'created_at': '2009-07-19'}, 'directors': {'id': 'villeneuve_denis'}}, {'movies': {'created_at': '2009-07-19'}, 'directors': {'id': 'villeneuve_denis'}}, does_not_raise(), id='test_incoming_legacy_state'), pytest.param([], defaultdict(dict, {}), does_not_raise(), id='test_empty_incoming_stream_state'), pytest.param(None, defaultdict(dict, {}), does_not_raise(), id='test_none_incoming_state'), pytest.param({}, defaultdict(dict, {}), does_not_raise(), id='test_empty_incoming_legacy_state'), pytest.param([{'type': 'NOT_REAL', 'stream': {'stream_state': {'created_at': '2009-07-19'}, 'stream_descriptor': {'name': 'movies', 'namespace': 'public'}}}], None, pytest.raises(ValidationError), id='test_invalid_stream_state_invalid_type'), pytest.param([{'type': 'STREAM', 'stream': {'stream_state': {'created_at': '2009-07-19'}}}], None, pytest.raises(ValidationError), id='test_invalid_stream_state_missing_descriptor'), pytest.param([{'type': 'GLOBAL', 'global': {'shared_state': {'shared_key': 'shared_val'}}}], None, pytest.raises(ValidationError), id='test_invalid_global_state_missing_streams'), pytest.param([{'type': 'GLOBAL', 'global': {'shared_state': {'shared_key': 'shared_val'}, 'stream_states': {'stream_state': {'created_at': '2009-07-19'}, 'stream_descriptor': {'name': 'movies', 'namespace': 'public'}}}}], None, pytest.raises(ValidationError), id='test_invalid_global_state_streams_not_list'), pytest.param([{'type': 'LEGACY', 'not': 'something'}], None, pytest.raises(ValueError), id='test_invalid_state_message_has_no_stream_global_or_data')])
def test_read_state(source, incoming_state, expected_state, expected_error):
    if False:
        return 10
    with tempfile.NamedTemporaryFile('w') as state_file:
        state_file.write(json.dumps(incoming_state))
        state_file.flush()
        with expected_error:
            actual = source.read_state(state_file.name)
            assert actual == expected_state

def test_read_invalid_state(source):
    if False:
        i = 10
        return i + 15
    with tempfile.NamedTemporaryFile('w') as state_file:
        state_file.write('invalid json content')
        state_file.flush()
        with pytest.raises(ValueError, match='Could not read json file'):
            source.read_state(state_file.name)

def test_read_state_sends_new_legacy_format_if_source_does_not_implement_read():
    if False:
        for i in range(10):
            print('nop')
    expected_state = [AirbyteStateMessage(type=AirbyteStateType.LEGACY, data={'movies': {'created_at': '2009-07-19'}, 'directors': {'id': 'villeneuve_denis'}})]
    source = MockAbstractSource()
    with tempfile.NamedTemporaryFile('w') as state_file:
        state_file.write(json.dumps({'movies': {'created_at': '2009-07-19'}, 'directors': {'id': 'villeneuve_denis'}}))
        state_file.flush()
        actual = source.read_state(state_file.name)
        assert actual == expected_state

@pytest.mark.parametrize('source, expected_state', [pytest.param(MockSource(), {}, id='test_source_implementing_read_returns_legacy_format'), pytest.param(MockAbstractSource(), [], id='test_source_not_implementing_read_returns_per_stream_format')])
def test_read_state_nonexistent(source, expected_state):
    if False:
        i = 10
        return i + 15
    assert source.read_state('') == expected_state

def test_read_catalog(source):
    if False:
        i = 10
        return i + 15
    configured_catalog = {'streams': [{'stream': {'name': 'mystream', 'json_schema': {'type': 'object', 'properties': {'k': 'v'}}, 'supported_sync_modes': ['full_refresh']}, 'destination_sync_mode': 'overwrite', 'sync_mode': 'full_refresh'}]}
    expected = ConfiguredAirbyteCatalog.parse_obj(configured_catalog)
    with tempfile.NamedTemporaryFile('w') as catalog_file:
        catalog_file.write(expected.json(exclude_unset=True))
        catalog_file.flush()
        actual = source.read_catalog(catalog_file.name)
        assert actual == expected

def test_internal_config(abstract_source, catalog):
    if False:
        return 10
    streams = abstract_source.streams(None)
    assert len(streams) == 2
    (http_stream, non_http_stream) = streams
    assert isinstance(http_stream, HttpStream)
    assert not isinstance(non_http_stream, HttpStream)
    http_stream.read_records.return_value = [{}] * 3
    non_http_stream.read_records.return_value = [{}] * 3
    logger = logging.getLogger(f"airbyte.{getattr(abstract_source, 'name', '')}")
    records = [r for r in abstract_source.read(logger=logger, config={}, catalog=catalog, state={})]
    assert len(records) == 3 + 3 + 3 + 3
    assert http_stream.read_records.called
    assert non_http_stream.read_records.called
    assert not http_stream.page_size
    assert not non_http_stream.page_size
    internal_config = {'some_config': 100, '_limit': 1}
    records = [r for r in abstract_source.read(logger=logger, config=internal_config, catalog=catalog, state={})]
    assert len(records) == 1 + 1 + 3 + 3
    assert '_limit' not in abstract_source.streams_config
    assert 'some_config' in abstract_source.streams_config
    internal_config = {'some_config': 100, '_limit': 20}
    records = [r for r in abstract_source.read(logger=logger, config=internal_config, catalog=catalog, state={})]
    assert len(records) == 3 + 3 + 3 + 3
    internal_config = {'some_config': 100, '_page_size': 2}
    records = [r for r in abstract_source.read(logger=logger, config=internal_config, catalog=catalog, state={})]
    assert '_page_size' not in abstract_source.streams_config
    assert 'some_config' in abstract_source.streams_config
    assert len(records) == 3 + 3 + 3 + 3
    assert http_stream.page_size == 2
    assert not non_http_stream.page_size

def test_internal_config_limit(mocker, abstract_source, catalog):
    if False:
        for i in range(10):
            print('nop')
    logger_mock = mocker.MagicMock()
    logger_mock.level = logging.DEBUG
    del catalog.streams[1]
    STREAM_LIMIT = 2
    SLICE_DEBUG_LOG_COUNT = 1
    FULL_RECORDS_NUMBER = 3
    TRACE_STATUS_COUNT = 3
    streams = abstract_source.streams(None)
    http_stream = streams[0]
    http_stream.read_records.return_value = [{}] * FULL_RECORDS_NUMBER
    internal_config = {'some_config': 100, '_limit': STREAM_LIMIT}
    catalog.streams[0].sync_mode = SyncMode.full_refresh
    records = [r for r in abstract_source.read(logger=logger_mock, config=internal_config, catalog=catalog, state={})]
    assert len(records) == STREAM_LIMIT + SLICE_DEBUG_LOG_COUNT + TRACE_STATUS_COUNT
    logger_info_args = [call[0][0] for call in logger_mock.info.call_args_list]
    read_log_record = [_l for _l in logger_info_args if _l.startswith('Read')]
    assert read_log_record[0].startswith(f'Read {STREAM_LIMIT} ')
    catalog.streams[0].sync_mode = SyncMode.incremental
    records = [r for r in abstract_source.read(logger=logger_mock, config={}, catalog=catalog, state={})]
    assert len(records) == FULL_RECORDS_NUMBER + SLICE_DEBUG_LOG_COUNT + TRACE_STATUS_COUNT + 1
    assert records[-2].type == Type.STATE
    assert records[-1].type == Type.TRACE
    logger_mock.reset_mock()
    records = [r for r in abstract_source.read(logger=logger_mock, config=internal_config, catalog=catalog, state={})]
    assert len(records) == STREAM_LIMIT + SLICE_DEBUG_LOG_COUNT + TRACE_STATUS_COUNT + 1
    assert records[-2].type == Type.STATE
    assert records[-1].type == Type.TRACE
    logger_info_args = [call[0][0] for call in logger_mock.info.call_args_list]
    read_log_record = [_l for _l in logger_info_args if _l.startswith('Read')]
    assert read_log_record[0].startswith(f'Read {STREAM_LIMIT} ')
SCHEMA = {'type': 'object', 'properties': {'value': {'type': 'string'}}}

def test_source_config_no_transform(mocker, abstract_source, catalog):
    if False:
        for i in range(10):
            print('nop')
    SLICE_DEBUG_LOG_COUNT = 1
    TRACE_STATUS_COUNT = 3
    logger_mock = mocker.MagicMock()
    logger_mock.level = logging.DEBUG
    streams = abstract_source.streams(None)
    (http_stream, non_http_stream) = streams
    http_stream.get_json_schema.return_value = non_http_stream.get_json_schema.return_value = SCHEMA
    (http_stream.read_records.return_value, non_http_stream.read_records.return_value) = [[{'value': 23}] * 5] * 2
    records = [r for r in abstract_source.read(logger=logger_mock, config={}, catalog=catalog, state={})]
    assert len(records) == 2 * (5 + SLICE_DEBUG_LOG_COUNT + TRACE_STATUS_COUNT)
    assert [r.record.data for r in records if r.type == Type.RECORD] == [{'value': 23}] * 2 * 5
    assert http_stream.get_json_schema.call_count == 5
    assert non_http_stream.get_json_schema.call_count == 5

def test_source_config_transform(mocker, abstract_source, catalog):
    if False:
        for i in range(10):
            print('nop')
    logger_mock = mocker.MagicMock()
    logger_mock.level = logging.DEBUG
    SLICE_DEBUG_LOG_COUNT = 2
    TRACE_STATUS_COUNT = 6
    streams = abstract_source.streams(None)
    (http_stream, non_http_stream) = streams
    http_stream.transformer = TypeTransformer(TransformConfig.DefaultSchemaNormalization)
    non_http_stream.transformer = TypeTransformer(TransformConfig.DefaultSchemaNormalization)
    http_stream.get_json_schema.return_value = non_http_stream.get_json_schema.return_value = SCHEMA
    (http_stream.read_records.return_value, non_http_stream.read_records.return_value) = ([{'value': 23}], [{'value': 23}])
    records = [r for r in abstract_source.read(logger=logger_mock, config={}, catalog=catalog, state={})]
    assert len(records) == 2 + SLICE_DEBUG_LOG_COUNT + TRACE_STATUS_COUNT
    assert [r.record.data for r in records if r.type == Type.RECORD] == [{'value': '23'}] * 2

def test_source_config_transform_and_no_transform(mocker, abstract_source, catalog):
    if False:
        for i in range(10):
            print('nop')
    logger_mock = mocker.MagicMock()
    logger_mock.level = logging.DEBUG
    SLICE_DEBUG_LOG_COUNT = 2
    TRACE_STATUS_COUNT = 6
    streams = abstract_source.streams(None)
    (http_stream, non_http_stream) = streams
    http_stream.transformer = TypeTransformer(TransformConfig.DefaultSchemaNormalization)
    http_stream.get_json_schema.return_value = non_http_stream.get_json_schema.return_value = SCHEMA
    (http_stream.read_records.return_value, non_http_stream.read_records.return_value) = ([{'value': 23}], [{'value': 23}])
    records = [r for r in abstract_source.read(logger=logger_mock, config={}, catalog=catalog, state={})]
    assert len(records) == 2 + SLICE_DEBUG_LOG_COUNT + TRACE_STATUS_COUNT
    assert [r.record.data for r in records if r.type == Type.RECORD] == [{'value': '23'}, {'value': 23}]

def test_read_default_http_availability_strategy_stream_available(catalog, mocker):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch.multiple(HttpStream, __abstractmethods__=set())
    mocker.patch.multiple(Stream, __abstractmethods__=set())

    class MockHttpStream(mocker.MagicMock, HttpStream):
        url_base = 'http://example.com'
        path = '/dummy/path'
        get_json_schema = mocker.MagicMock()

        def supports_incremental(self):
            if False:
                while True:
                    i = 10
            return True

        def __init__(self, *args, **kvargs):
            if False:
                print('Hello World!')
            mocker.MagicMock.__init__(self)
            HttpStream.__init__(self, *args, kvargs)
            self.read_records = mocker.MagicMock()

    class MockStream(mocker.MagicMock, Stream):
        page_size = None
        get_json_schema = mocker.MagicMock()

        def __init__(self, *args, **kvargs):
            if False:
                for i in range(10):
                    print('nop')
            mocker.MagicMock.__init__(self)
            self.read_records = mocker.MagicMock()
    streams = [MockHttpStream(), MockStream()]
    (http_stream, non_http_stream) = streams
    assert isinstance(http_stream, HttpStream)
    assert not isinstance(non_http_stream, HttpStream)
    assert isinstance(http_stream.availability_strategy, HttpAvailabilityStrategy)
    assert non_http_stream.availability_strategy is None
    http_stream.read_records.return_value = iter([{'value': 'test'}] + [{}] * 3)
    non_http_stream.read_records.return_value = iter([{}] * 3)
    source = MockAbstractSource(streams=streams)
    logger = logging.getLogger(f"airbyte.{getattr(abstract_source, 'name', '')}")
    records = [r for r in source.read(logger=logger, config={}, catalog=catalog, state={})]
    assert len(records) == 3 + 3 + 3 + 3
    assert http_stream.read_records.called
    assert non_http_stream.read_records.called

def test_read_default_http_availability_strategy_stream_unavailable(catalog, mocker, caplog):
    if False:
        while True:
            i = 10
    mocker.patch.multiple(Stream, __abstractmethods__=set())

    class MockHttpStream(HttpStream):
        url_base = 'https://test_base_url.com'
        primary_key = ''

        def __init__(self, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)
            self.resp_counter = 1

        def next_page_token(self, response: requests.Response) -> Optional[Mapping[str, Any]]:
            if False:
                while True:
                    i = 10
            return None

        def path(self, **kwargs) -> str:
            if False:
                while True:
                    i = 10
            return ''

        def parse_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
            if False:
                i = 10
                return i + 15
            stub_response = {'data': self.resp_counter}
            self.resp_counter += 1
            yield stub_response

    class MockStream(mocker.MagicMock, Stream):
        page_size = None
        get_json_schema = mocker.MagicMock()

        def __init__(self, *args, **kvargs):
            if False:
                i = 10
                return i + 15
            mocker.MagicMock.__init__(self)
            self.read_records = mocker.MagicMock()
    streams = [MockHttpStream(), MockStream()]
    (http_stream, non_http_stream) = streams
    assert isinstance(http_stream, HttpStream)
    assert not isinstance(non_http_stream, HttpStream)
    assert isinstance(http_stream.availability_strategy, HttpAvailabilityStrategy)
    assert non_http_stream.availability_strategy is None
    non_http_stream.read_records.return_value = iter([{}] * 3)
    req = requests.Response()
    req.status_code = 403
    mocker.patch.object(requests.Session, 'send', return_value=req)
    source = MockAbstractSource(streams=streams)
    logger = logging.getLogger('test_read_default_http_availability_strategy_stream_unavailable')
    with caplog.at_level(logging.WARNING):
        records = [r for r in source.read(logger=logger, config={}, catalog=catalog, state={})]
    assert len(records) == 0 + 3 + 3
    assert non_http_stream.read_records.called
    expected_logs = [f"Skipped syncing stream '{http_stream.name}' because it was unavailable.", f'Unable to read {http_stream.name} stream.', 'This is most likely due to insufficient permissions on the credentials in use.', f'Please visit https://docs.airbyte.com/integrations/sources/{source.name} to learn more.']
    for message in expected_logs:
        assert message in caplog.text

def test_read_default_http_availability_strategy_parent_stream_unavailable(catalog, mocker, caplog):
    if False:
        i = 10
        return i + 15
    'Test default availability strategy if error happens during slice extraction (reading of parent stream)'
    mocker.patch.multiple(Stream, __abstractmethods__=set())

    class MockHttpParentStream(HttpStream):
        url_base = 'https://test_base_url.com'
        primary_key = ''

        def __init__(self, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)
            self.resp_counter = 1

        def next_page_token(self, response: requests.Response) -> Optional[Mapping[str, Any]]:
            if False:
                return 10
            return None

        def path(self, **kwargs) -> str:
            if False:
                i = 10
                return i + 15
            return ''

        def parse_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
            if False:
                i = 10
                return i + 15
            stub_response = {'data': self.resp_counter}
            self.resp_counter += 1
            yield stub_response

    class MockHttpStream(HttpSubStream):
        url_base = 'https://test_base_url.com'
        primary_key = ''

        def __init__(self, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)
            self.resp_counter = 1

        def next_page_token(self, response: requests.Response) -> Optional[Mapping[str, Any]]:
            if False:
                print('Hello World!')
            return None

        def path(self, **kwargs) -> str:
            if False:
                while True:
                    i = 10
            return ''

        def parse_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
            if False:
                print('Hello World!')
            stub_response = {'data': self.resp_counter}
            self.resp_counter += 1
            yield stub_response
    http_stream = MockHttpStream(parent=MockHttpParentStream())
    streams = [http_stream]
    assert isinstance(http_stream, HttpSubStream)
    assert isinstance(http_stream.availability_strategy, HttpAvailabilityStrategy)
    req = requests.Response()
    req.status_code = 403
    mocker.patch.object(requests.Session, 'send', return_value=req)
    source = MockAbstractSource(streams=streams)
    logger = logging.getLogger('test_read_default_http_availability_strategy_parent_stream_unavailable')
    configured_catalog = {'streams': [{'stream': {'name': 'mock_http_stream', 'json_schema': {'type': 'object', 'properties': {'k': 'v'}}, 'supported_sync_modes': ['full_refresh']}, 'destination_sync_mode': 'overwrite', 'sync_mode': 'full_refresh'}]}
    catalog = ConfiguredAirbyteCatalog.parse_obj(configured_catalog)
    with caplog.at_level(logging.WARNING):
        records = [r for r in source.read(logger=logger, config={}, catalog=catalog, state={})]
    assert len(records) == 0
    expected_logs = [f"Skipped syncing stream '{http_stream.name}' because it was unavailable.", f'Unable to get slices for {http_stream.name} stream, because of error in parent stream', 'This is most likely due to insufficient permissions on the credentials in use.', f'Please visit https://docs.airbyte.com/integrations/sources/{source.name} to learn more.']
    for message in expected_logs:
        assert message in caplog.text