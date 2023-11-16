import warnings
import pytest
from prefect.deprecated.data_documents import _SERIALIZERS, DataDocument, Serializer, register_serializer
from prefect.settings import PREFECT_ASYNC_FETCH_STATE_RESULT, temporary_settings
from prefect.states import Completed

@pytest.fixture(autouse=True)
def reset_registered_serializers(monkeypatch):
    if False:
        while True:
            i = 10
    _copy = _SERIALIZERS.copy()
    yield
    _SERIALIZERS.clear()
    _SERIALIZERS.update(_copy)

@pytest.fixture(autouse=True)
def ignore_deprecation_warnings():
    if False:
        return 10
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        yield

class TestDataDocument:

    def test_encode_does_not_allow_unknown_encoding(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError, match="Unregistered encoding 'foo'"):
            DataDocument.encode(encoding='foo', data='test')

    def test_encode_requires_serializer_to_implement_dumps(self):
        if False:
            for i in range(10):
                print('nop')

        @register_serializer('foo')
        class BadSerializer(Serializer):
            pass
        with pytest.raises(NotImplementedError):
            DataDocument.encode(encoding='foo', data='test')

    def test_encode_serializes_data_using_dispatcher(self):
        if False:
            return 10

        @register_serializer('foo')
        class FooSerializer(DataDocument):

            @staticmethod
            def dumps(data):
                if False:
                    print('Hello World!')
                return (data + 'foo').encode()
        result = DataDocument.encode(encoding='foo', data='test')
        assert result.encoding == 'foo'
        assert result.blob == b'testfoo'

    def test_decode_requires_serializer_to_implement_loads(self):
        if False:
            while True:
                i = 10

        @register_serializer('foo')
        class BadSerializer(Serializer):
            pass
        datadoc = DataDocument(encoding='foo', blob=b'test')
        with pytest.raises(NotImplementedError):
            datadoc.decode()

    def test_encode_deserializes_data_using_dispatcher(self):
        if False:
            return 10

        @register_serializer('foo')
        class FooSerializer(DataDocument):

            @staticmethod
            def loads(blob):
                if False:
                    i = 10
                    return i + 15
                return blob.decode() + 'foo'
        datadoc = DataDocument(encoding='foo', blob=b'test')
        assert datadoc.decode() == 'testfoo'

async def test_async_state_result_with_data_document():
    state = Completed(data=DataDocument.encode('json', 1))
    result = state.result(fetch=False)
    with temporary_settings({PREFECT_ASYNC_FETCH_STATE_RESULT: False}):
        with pytest.warns(DeprecationWarning, match='State.result\\(\\) was called from an async context but not awaited.'):
            result = state.result()
    assert result == 1

async def test_async_state_result_does_not_raise_warning_with_opt_out():
    state = Completed(data=DataDocument.encode('json', 1))
    with temporary_settings({PREFECT_ASYNC_FETCH_STATE_RESULT: False}):
        result = state.result(fetch=False)
    assert result == 1