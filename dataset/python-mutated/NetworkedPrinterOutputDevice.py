from UM.FileHandler.FileHandler import FileHandler
from UM.Logger import Logger
from UM.Scene.SceneNode import SceneNode
from cura.API import Account
from cura.CuraApplication import CuraApplication
from cura.PrinterOutput.PrinterOutputDevice import PrinterOutputDevice, ConnectionState, ConnectionType
from PyQt6.QtNetwork import QHttpMultiPart, QHttpPart, QNetworkRequest, QNetworkAccessManager, QNetworkReply, QAuthenticator
from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QObject, QUrl, QCoreApplication
from time import time
from typing import Callable, Dict, List, Optional, Union
from enum import IntEnum
import os
import gzip
from cura.Settings.CuraContainerRegistry import CuraContainerRegistry

class AuthState(IntEnum):
    NotAuthenticated = 1
    AuthenticationRequested = 2
    Authenticated = 3
    AuthenticationDenied = 4
    AuthenticationReceived = 5

class NetworkedPrinterOutputDevice(PrinterOutputDevice):
    authenticationStateChanged = pyqtSignal()

    def __init__(self, device_id, address: str, properties: Dict[bytes, bytes], connection_type: ConnectionType=ConnectionType.NetworkConnection, parent: QObject=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(device_id=device_id, connection_type=connection_type, parent=parent)
        self._manager = None
        self._timeout_time = 10
        self._last_response_time = None
        self._last_request_time = None
        self._api_prefix = ''
        self._address = address
        self._properties = properties
        self._user_agent = '%s/%s ' % (CuraApplication.getInstance().getApplicationName(), CuraApplication.getInstance().getVersion())
        self._onFinishedCallbacks = {}
        self._authentication_state = AuthState.NotAuthenticated
        self._kept_alive_multiparts = {}
        self._sending_gcode = False
        self._compressing_gcode = False
        self._gcode = []
        self._connection_state_before_timeout = None

    def requestWrite(self, nodes: List['SceneNode'], file_name: Optional[str]=None, limit_mimetypes: bool=False, file_handler: Optional['FileHandler']=None, filter_by_machine: bool=False, **kwargs) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError('requestWrite needs to be implemented')

    def setAuthenticationState(self, authentication_state: AuthState) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._authentication_state != authentication_state:
            self._authentication_state = authentication_state
            self.authenticationStateChanged.emit()

    @pyqtProperty(int, notify=authenticationStateChanged)
    def authenticationState(self) -> AuthState:
        if False:
            while True:
                i = 10
        return self._authentication_state

    def _compressDataAndNotifyQt(self, data_to_append: str) -> bytes:
        if False:
            while True:
                i = 10
        compressed_data = gzip.compress(data_to_append.encode('utf-8'))
        self._progress_message.setProgress(-1)
        QCoreApplication.processEvents()
        self._last_response_time = time()
        return compressed_data

    def _compressGCode(self) -> Optional[bytes]:
        if False:
            for i in range(10):
                print('nop')
        self._compressing_gcode = True
        max_chars_per_line = int(1024 * 1024 / 4)
        'Mash the data into single string'
        file_data_bytes_list = []
        batched_lines = []
        batched_lines_count = 0
        for line in self._gcode:
            if not self._compressing_gcode:
                self._progress_message.hide()
                return None
            batched_lines.append(line)
            batched_lines_count += len(line)
            if batched_lines_count >= max_chars_per_line:
                file_data_bytes_list.append(self._compressDataAndNotifyQt(''.join(batched_lines)))
                batched_lines = []
                batched_lines_count = 0
        if len(batched_lines) != 0:
            file_data_bytes_list.append(self._compressDataAndNotifyQt(''.join(batched_lines)))
        self._compressing_gcode = False
        return b''.join(file_data_bytes_list)

    def _update(self) -> None:
        if False:
            return 10
        '\n        Update the connection state of this device.\n\n        This is called on regular intervals.\n        '
        if self._last_response_time:
            time_since_last_response = time() - self._last_response_time
        else:
            time_since_last_response = 0
        if self._last_request_time:
            time_since_last_request = time() - self._last_request_time
        else:
            time_since_last_request = float('inf')
        if time_since_last_response > self._timeout_time >= time_since_last_request:
            if self._connection_state_before_timeout is None:
                self._connection_state_before_timeout = self.connectionState
            self.setConnectionState(ConnectionState.Closed)
        elif self.connectionState == ConnectionState.Closed:
            if self._connection_state_before_timeout is not None:
                self.setConnectionState(self._connection_state_before_timeout)
                self._connection_state_before_timeout = None

    def _createEmptyRequest(self, target: str, content_type: Optional[str]='application/json') -> QNetworkRequest:
        if False:
            print('Hello World!')
        url = QUrl('http://' + self._address + self._api_prefix + target)
        request = QNetworkRequest(url)
        if content_type is not None:
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, content_type)
        request.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, self._user_agent)
        return request

    def createFormPart(self, content_header: str, data: bytes, content_type: Optional[str]=None) -> QHttpPart:
        if False:
            print('Hello World!')
        'This method was only available privately before, but it was actually called from SendMaterialJob.py.\n\n        We now have a public equivalent as well. We did not remove the private one as plugins might be using that.\n        '
        return self._createFormPart(content_header, data, content_type)

    def _createFormPart(self, content_header: str, data: bytes, content_type: Optional[str]=None) -> QHttpPart:
        if False:
            i = 10
            return i + 15
        part = QHttpPart()
        if not content_header.startswith('form-data;'):
            content_header = 'form-data; ' + content_header
        part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, content_header)
        if content_type is not None:
            part.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, content_type)
        part.setBody(data)
        return part

    def _getUserName(self) -> str:
        if False:
            i = 10
            return i + 15
        'Convenience function to get the username, either from the cloud or from the OS.'
        account = CuraApplication.getInstance().getCuraAPI().account
        if account and account.isLoggedIn:
            return account.userName
        for name in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
            user = os.environ.get(name)
            if user:
                return user
        return 'Unknown User'

    def _clearCachedMultiPart(self, reply: QNetworkReply) -> None:
        if False:
            while True:
                i = 10
        if reply in self._kept_alive_multiparts:
            del self._kept_alive_multiparts[reply]

    def _validateManager(self) -> None:
        if False:
            return 10
        if self._manager is None:
            self._createNetworkManager()
        assert self._manager is not None

    def put(self, url: str, data: Union[str, bytes], content_type: Optional[str]='application/json', on_finished: Optional[Callable[[QNetworkReply], None]]=None, on_progress: Optional[Callable[[int, int], None]]=None) -> None:
        if False:
            while True:
                i = 10
        'Sends a put request to the given path.\n\n        :param url: The path after the API prefix.\n        :param data: The data to be sent in the body\n        :param content_type: The content type of the body data.\n        :param on_finished: The function to call when the response is received.\n        :param on_progress: The function to call when the progress changes. Parameters are bytes_sent / bytes_total.\n        '
        self._validateManager()
        request = self._createEmptyRequest(url, content_type=content_type)
        self._last_request_time = time()
        if not self._manager:
            Logger.log('e', 'No network manager was created to execute the PUT call with.')
            return
        body = data if isinstance(data, bytes) else data.encode()
        reply = self._manager.put(request, body)
        self._registerOnFinishedCallback(reply, on_finished)
        if on_progress is not None:
            reply.uploadProgress.connect(on_progress)

    def delete(self, url: str, on_finished: Optional[Callable[[QNetworkReply], None]]) -> None:
        if False:
            while True:
                i = 10
        'Sends a delete request to the given path.\n\n        :param url: The path after the API prefix.\n        :param on_finished: The function to be call when the response is received.\n        '
        self._validateManager()
        request = self._createEmptyRequest(url)
        self._last_request_time = time()
        if not self._manager:
            Logger.log('e', 'No network manager was created to execute the DELETE call with.')
            return
        reply = self._manager.deleteResource(request)
        self._registerOnFinishedCallback(reply, on_finished)

    def get(self, url: str, on_finished: Optional[Callable[[QNetworkReply], None]]) -> None:
        if False:
            i = 10
            return i + 15
        'Sends a get request to the given path.\n\n        :param url: The path after the API prefix.\n        :param on_finished: The function to be call when the response is received.\n        '
        self._validateManager()
        request = self._createEmptyRequest(url)
        self._last_request_time = time()
        if not self._manager:
            Logger.log('e', 'No network manager was created to execute the GET call with.')
            return
        reply = self._manager.get(request)
        self._registerOnFinishedCallback(reply, on_finished)

    def post(self, url: str, data: Union[str, bytes], on_finished: Optional[Callable[[QNetworkReply], None]], on_progress: Optional[Callable[[int, int], None]]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Sends a post request to the given path.\n\n        :param url: The path after the API prefix.\n        :param data: The data to be sent in the body\n        :param on_finished: The function to call when the response is received.\n        :param on_progress: The function to call when the progress changes. Parameters are bytes_sent / bytes_total.\n        '
        self._validateManager()
        request = self._createEmptyRequest(url)
        self._last_request_time = time()
        if not self._manager:
            Logger.log('e', 'Could not find manager.')
            return
        body = data if isinstance(data, bytes) else data.encode()
        reply = self._manager.post(request, body)
        if on_progress is not None:
            reply.uploadProgress.connect(on_progress)
        self._registerOnFinishedCallback(reply, on_finished)

    def postFormWithParts(self, target: str, parts: List[QHttpPart], on_finished: Optional[Callable[[QNetworkReply], None]], on_progress: Optional[Callable[[int, int], None]]=None) -> QNetworkReply:
        if False:
            for i in range(10):
                print('nop')
        self._validateManager()
        request = self._createEmptyRequest(target, content_type=None)
        multi_post_part = QHttpMultiPart(QHttpMultiPart.ContentType.FormDataType)
        for part in parts:
            multi_post_part.append(part)
        self._last_request_time = time()
        if self._manager is not None:
            reply = self._manager.post(request, multi_post_part)
            self._kept_alive_multiparts[reply] = multi_post_part
            if on_progress is not None:
                reply.uploadProgress.connect(on_progress)
            self._registerOnFinishedCallback(reply, on_finished)
            return reply
        else:
            Logger.log('e', 'Could not find manager.')

    def postForm(self, target: str, header_data: str, body_data: bytes, on_finished: Optional[Callable[[QNetworkReply], None]], on_progress: Callable=None) -> None:
        if False:
            i = 10
            return i + 15
        post_part = QHttpPart()
        post_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, header_data)
        post_part.setBody(body_data)
        self.postFormWithParts(target, [post_part], on_finished, on_progress)

    def _onAuthenticationRequired(self, reply: QNetworkReply, authenticator: QAuthenticator) -> None:
        if False:
            return 10
        Logger.log('w', 'Request to {url} required authentication, which was not implemented'.format(url=reply.url().toString()))

    def _createNetworkManager(self) -> None:
        if False:
            print('Hello World!')
        Logger.log('d', 'Creating network manager')
        if self._manager:
            self._manager.finished.disconnect(self._handleOnFinished)
            self._manager.authenticationRequired.disconnect(self._onAuthenticationRequired)
        self._manager = QNetworkAccessManager()
        self._manager.finished.connect(self._handleOnFinished)
        self._manager.authenticationRequired.connect(self._onAuthenticationRequired)
        if self._properties.get(b'temporary', b'false') != b'true':
            self._checkCorrectGroupName(self.getId(), self.name)

    def _registerOnFinishedCallback(self, reply: QNetworkReply, on_finished: Optional[Callable[[QNetworkReply], None]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if on_finished is not None:
            self._onFinishedCallbacks[reply.url().toString() + str(reply.operation())] = on_finished

    def _checkCorrectGroupName(self, device_id: str, group_name: str) -> None:
        if False:
            while True:
                i = 10
        'This method checks if the name of the group stored in the definition container is correct.\n\n        After updating from 3.2 to 3.3 some group names may be temporary. If there is a mismatch in the name of the group\n        then all the container stacks are updated, both the current and the hidden ones.\n        '
        global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        active_machine_network_name = CuraApplication.getInstance().getMachineManager().activeMachineNetworkKey()
        if global_container_stack and device_id == active_machine_network_name:
            if CuraApplication.getInstance().getMachineManager().activeMachineNetworkGroupName != group_name:
                metadata_filter = {'um_network_key': active_machine_network_name}
                containers = CuraContainerRegistry.getInstance().findContainerStacks(type='machine', **metadata_filter)
                for container in containers:
                    container.setMetaDataEntry('group_name', group_name)

    def _handleOnFinished(self, reply: QNetworkReply) -> None:
        if False:
            return 10
        if reply.operation() == QNetworkAccessManager.Operation.PostOperation:
            self._clearCachedMultiPart(reply)
        if reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) is None:
            return
        self._last_response_time = time()
        if self.connectionState == ConnectionState.Connecting:
            self.setConnectionState(ConnectionState.Connected)
        callback_key = reply.url().toString() + str(reply.operation())
        try:
            if callback_key in self._onFinishedCallbacks:
                self._onFinishedCallbacks[callback_key](reply)
        except Exception:
            Logger.logException('w', 'something went wrong with callback')

    @pyqtSlot(str, result=str)
    def getProperty(self, key: str) -> str:
        if False:
            while True:
                i = 10
        bytes_key = key.encode('utf-8')
        if bytes_key in self._properties:
            return self._properties.get(bytes_key, b'').decode('utf-8')
        else:
            return ''

    def getProperties(self):
        if False:
            for i in range(10):
                print('nop')
        return self._properties

    @pyqtProperty(str, constant=True)
    def key(self) -> str:
        if False:
            print('Hello World!')
        'Get the unique key of this machine\n\n        :return: key String containing the key of the machine.\n        '
        return self._id

    @pyqtProperty(str, constant=True)
    def address(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The IP address of the printer.'
        return self._properties.get(b'address', b'').decode('utf-8')

    @pyqtProperty(str, constant=True)
    def name(self) -> str:
        if False:
            while True:
                i = 10
        'Name of the printer (as returned from the ZeroConf properties)'
        return self._properties.get(b'name', b'').decode('utf-8')

    @pyqtProperty(str, constant=True)
    def firmwareVersion(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Firmware version (as returned from the ZeroConf properties)'
        return self._properties.get(b'firmware_version', b'').decode('utf-8')

    @pyqtProperty(str, constant=True)
    def printerType(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return NetworkedPrinterOutputDevice.applyPrinterTypeMapping(self._properties.get(b'printer_type', b'Unknown').decode('utf-8'))

    @staticmethod
    def applyPrinterTypeMapping(printer_type):
        if False:
            print('Hello World!')
        _PRINTER_TYPE_NAME = {'fire_e': 'ultimaker_method', 'lava_f': 'ultimaker_methodx', 'magma_10': 'ultimaker_methodxl'}
        if printer_type in _PRINTER_TYPE_NAME:
            return _PRINTER_TYPE_NAME[printer_type]
        return printer_type

    @pyqtProperty(str, constant=True)
    def ipAddress(self) -> str:
        if False:
            return 10
        'IP address of this printer'
        return self._address