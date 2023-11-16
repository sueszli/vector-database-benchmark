import time
from unittest.mock import MagicMock, patch
from PyQt6.QtNetwork import QNetworkAccessManager
from PyQt6.QtCore import QUrl
from cura.PrinterOutput.NetworkedPrinterOutputDevice import NetworkedPrinterOutputDevice, AuthState
from cura.PrinterOutput.PrinterOutputDevice import ConnectionState

def test_properties():
    if False:
        print('Hello World!')
    properties = {b'firmware_version': b'12', b'printer_type': b'BHDHAHHADAD', b'address': b'ZOMG', b'name': b':(', b'testProp': b'zomg'}
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties=properties)
    assert output_device.address == 'ZOMG'
    assert output_device.firmwareVersion == '12'
    assert output_device.printerType == 'BHDHAHHADAD'
    assert output_device.ipAddress == '127.0.0.1'
    assert output_device.name == ':('
    assert output_device.key == 'test'
    assert output_device.getProperties() == properties
    assert output_device.getProperty('testProp') == 'zomg'
    assert output_device.getProperty('whateverr') == ''

def test_authenticationState():
    if False:
        i = 10
        return i + 15
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties={})
    output_device.setAuthenticationState(AuthState.Authenticated)
    assert output_device.authenticationState == AuthState.Authenticated

def test_post():
    if False:
        return 10
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties={})
    mocked_network_manager = MagicMock()
    output_device._manager = mocked_network_manager
    reply = MagicMock()
    reply.operation = MagicMock(return_value=QNetworkAccessManager.Operation.PostOperation)
    reply.url = MagicMock(return_value=QUrl('127.0.0.1'))
    mocked_network_manager.post = MagicMock(return_value=reply)
    mocked_callback_handler = MagicMock()
    output_device.post('whatever', 'omgzomg', on_finished=mocked_callback_handler.onFinished)
    output_device._handleOnFinished(reply)
    mocked_callback_handler.onFinished.assert_called_once_with(reply)

def test_get():
    if False:
        i = 10
        return i + 15
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties={})
    mocked_network_manager = MagicMock()
    output_device._manager = mocked_network_manager
    reply = MagicMock()
    reply.operation = MagicMock(return_value=QNetworkAccessManager.Operation.PostOperation)
    reply.url = MagicMock(return_value=QUrl('127.0.0.1'))
    mocked_network_manager.get = MagicMock(return_value=reply)
    mocked_callback_handler = MagicMock()
    output_device.get('whatever', on_finished=mocked_callback_handler.onFinished)
    output_device._handleOnFinished(reply)
    mocked_callback_handler.onFinished.assert_called_once_with(reply)

def test_delete():
    if False:
        for i in range(10):
            print('nop')
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties={})
    mocked_network_manager = MagicMock()
    output_device._manager = mocked_network_manager
    reply = MagicMock()
    reply.operation = MagicMock(return_value=QNetworkAccessManager.Operation.PostOperation)
    reply.url = MagicMock(return_value=QUrl('127.0.0.1'))
    mocked_network_manager.deleteResource = MagicMock(return_value=reply)
    mocked_callback_handler = MagicMock()
    output_device.delete('whatever', on_finished=mocked_callback_handler.onFinished)
    output_device._handleOnFinished(reply)
    mocked_callback_handler.onFinished.assert_called_once_with(reply)

def test_put():
    if False:
        while True:
            i = 10
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties={})
    mocked_network_manager = MagicMock()
    output_device._manager = mocked_network_manager
    reply = MagicMock()
    reply.operation = MagicMock(return_value=QNetworkAccessManager.Operation.PostOperation)
    reply.url = MagicMock(return_value=QUrl('127.0.0.1'))
    mocked_network_manager.put = MagicMock(return_value=reply)
    mocked_callback_handler = MagicMock()
    output_device.put('whatever', 'omgzomg', on_finished=mocked_callback_handler.onFinished)
    output_device._handleOnFinished(reply)
    mocked_callback_handler.onFinished.assert_called_once_with(reply)

def test_timeout():
    if False:
        i = 10
        return i + 15
    with patch('UM.Qt.QtApplication.QtApplication.getInstance'):
        output_device = NetworkedPrinterOutputDevice(device_id='test', address='127.0.0.1', properties={})
    with patch('cura.CuraApplication.CuraApplication.getInstance'):
        output_device.setConnectionState(ConnectionState.Connected)
    assert output_device.connectionState == ConnectionState.Connected
    output_device._update()
    output_device._last_response_time = time.time() - 15
    output_device._last_request_time = time.time() - 5
    with patch('cura.CuraApplication.CuraApplication.getInstance'):
        output_device._update()
    assert output_device.connectionState == ConnectionState.Closed