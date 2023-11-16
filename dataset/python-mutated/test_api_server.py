from pathlib import Path
from aiogram.client.telegram import PRODUCTION, BareFilesPathWrapper, SimpleFilesPathWrapper, TelegramAPIServer

class TestAPIServer:

    def test_method_url(self):
        if False:
            return 10
        method_url = PRODUCTION.api_url(token='42:TEST', method='apiMethod')
        assert method_url == 'https://api.telegram.org/bot42:TEST/apiMethod'

    def test_file_url(self):
        if False:
            return 10
        file_url = PRODUCTION.file_url(token='42:TEST', path='path')
        assert file_url == 'https://api.telegram.org/file/bot42:TEST/path'

    def test_from_base(self):
        if False:
            while True:
                i = 10
        local_server = TelegramAPIServer.from_base('http://localhost:8081', is_local=True)
        method_url = local_server.api_url('42:TEST', method='apiMethod')
        file_url = local_server.file_url(token='42:TEST', path='path')
        assert method_url == 'http://localhost:8081/bot42:TEST/apiMethod'
        assert file_url == 'http://localhost:8081/file/bot42:TEST/path'
        assert local_server.is_local

class TestBareFilesPathWrapper:

    def test_to_local(self):
        if False:
            while True:
                i = 10
        wrapper = BareFilesPathWrapper()
        assert wrapper.to_local('/path/to/file.dat') == '/path/to/file.dat'

    def test_to_server(self):
        if False:
            while True:
                i = 10
        wrapper = BareFilesPathWrapper()
        assert wrapper.to_server('/path/to/file.dat') == '/path/to/file.dat'

class TestSimpleFilesPathWrapper:

    def test_to_local(self):
        if False:
            for i in range(10):
                print('nop')
        wrapper = SimpleFilesPathWrapper(Path('/etc/telegram-bot-api/data'), Path('/opt/app/data'))
        assert wrapper.to_local('/etc/telegram-bot-api/data/documents/file.dat') == Path('/opt/app/data/documents/file.dat')

    def test_to_server(self):
        if False:
            return 10
        wrapper = SimpleFilesPathWrapper(Path('/etc/telegram-bot-api/data'), Path('/opt/app/data'))
        assert wrapper.to_server('/opt/app/data/documents/file.dat') == Path('/etc/telegram-bot-api/data/documents/file.dat')