from unittest import mock
import pytest
from ulauncher.api.client.Client import Client
from ulauncher.api.extension import Extension

class TestClient:

    @pytest.fixture(autouse=True)
    def sock_client(self, mocker):
        if False:
            i = 10
            return i + 15
        return mocker.patch('ulauncher.api.client.Client.Gio.SocketClient')

    @pytest.fixture(autouse=True)
    def mainloop(self, mocker):
        if False:
            return 10
        return mocker.patch('ulauncher.api.client.Client.GLib.MainLoop.new')

    @pytest.fixture(autouse=True)
    def framer(self, mocker):
        if False:
            print('Hello World!')
        return mocker.patch('ulauncher.api.client.Client.JSONFramer')

    @pytest.fixture(autouse=True)
    def timer(self, mocker):
        if False:
            for i in range(10):
                print('nop')
        return mocker.patch('ulauncher.api.client.Client.timer')

    @pytest.fixture
    def extension(self):
        if False:
            return 10
        ext = mock.create_autospec(Extension)
        ext.extension_id = 'com.example.test-extension'
        return ext

    @pytest.fixture
    def client(self, extension, framer, sock_client):
        if False:
            for i in range(10):
                print('nop')
        client = Client(extension)
        client.framer = framer
        client.client = sock_client
        return client

    def test_connect__connect_is_called(self, client, mainloop):
        if False:
            print('Hello World!')
        client.connect()
        client.client.connect.assert_called_once()
        client.framer.send.assert_called_once()
        mainloop.return_value.run.assert_called_once()

    def test_on_message__trigger_event__is_called(self, client, extension):
        if False:
            i = 10
            return i + 15
        client.on_message(mock.Mock(), {'hello': 'world'})
        extension.trigger_event.assert_called_with({'hello': 'world'})

    def test_on_close__UnloadEvent__is_triggered(self, client, extension):
        if False:
            while True:
                i = 10
        client.on_close(mock.Mock())
        extension.trigger_event.assert_called_with({'type': 'event:unload'})

    def test_send__ws_send__is_called(self, client):
        if False:
            return 10
        client.send({'hello': 'world'})
        client.framer.send.assert_called_with({'hello': 'world'})