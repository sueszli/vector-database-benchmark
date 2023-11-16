from pytest import CaptureFixture
from structlog.processors import JSONRenderer
from structlog.types import BindableLogger
from litestar.logging.config import StructLoggingConfig, default_json_serializer
from litestar.serialization import decode_json
from litestar.testing import create_test_client

def test_structlog_config_default(capsys: CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    with create_test_client([], logging_config=StructLoggingConfig()) as client:
        assert client.app.logger
        assert isinstance(client.app.logger.bind(), BindableLogger)
        client.app.logger.info('message', key='value')
        log_messages = [decode_json(value=x) for x in capsys.readouterr().out.splitlines()]
        assert len(log_messages) == 1
        log_messages[0].pop('timestamp')
        assert log_messages[0] == {'event': 'message', 'key': 'value', 'level': 'info'}

def test_structlog_config_specify_processors(capsys: CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    logging_config = StructLoggingConfig(processors=[JSONRenderer(serializer=default_json_serializer)])
    with create_test_client([], logging_config=logging_config) as client:
        assert client.app.logger
        assert isinstance(client.app.logger.bind(), BindableLogger)
        client.app.logger.info('message1', key='value1')
        client.app.logger.info('message2', key='value2')
        log_messages = [decode_json(value=x) for x in capsys.readouterr().out.splitlines()]
        assert log_messages == [{'key': 'value1', 'event': 'message1'}, {'key': 'value2', 'event': 'message2'}]