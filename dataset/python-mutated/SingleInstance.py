import json
import os
from typing import List, Optional
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from UM.Qt.QtApplication import QtApplication
from UM.Logger import Logger

class SingleInstance:

    def __init__(self, application: QtApplication, files_to_open: Optional[List[str]]) -> None:
        if False:
            i = 10
            return i + 15
        self._application = application
        self._files_to_open = files_to_open
        self._single_instance_server = None
        self._application.getPreferences().addPreference('cura/single_instance_clear_before_load', True)

    def startClient(self) -> bool:
        if False:
            print('Hello World!')
        Logger.log('i', 'Checking for the presence of an ready running Cura instance.')
        single_instance_socket = QLocalSocket(self._application)
        Logger.log('d', 'Full single instance server name: %s', single_instance_socket.fullServerName())
        single_instance_socket.connectToServer('ultimaker-cura')
        single_instance_socket.waitForConnected(msecs=3000)
        if single_instance_socket.state() != QLocalSocket.LocalSocketState.ConnectedState:
            return False
        if not self._files_to_open:
            Logger.log('i', 'No file need to be opened, do nothing.')
            return True
        if single_instance_socket.state() == QLocalSocket.LocalSocketState.ConnectedState:
            Logger.log('i', 'Connection has been made to the single-instance Cura socket.')
            if self._application.getPreferences().getValue('cura/single_instance_clear_before_load'):
                payload = {'command': 'clear-all'}
                single_instance_socket.write(bytes(json.dumps(payload) + '\n', encoding='ascii'))
            payload = {'command': 'focus'}
            single_instance_socket.write(bytes(json.dumps(payload) + '\n', encoding='ascii'))
            for filename in self._files_to_open:
                payload = {'command': 'open', 'filePath': os.path.abspath(filename)}
                single_instance_socket.write(bytes(json.dumps(payload) + '\n', encoding='ascii'))
            payload = {'command': 'close-connection'}
            single_instance_socket.write(bytes(json.dumps(payload) + '\n', encoding='ascii'))
            single_instance_socket.flush()
            single_instance_socket.waitForDisconnected()
        return True

    def startServer(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._single_instance_server = QLocalServer()
        if self._single_instance_server:
            self._single_instance_server.newConnection.connect(self._onClientConnected)
            self._single_instance_server.listen('ultimaker-cura')
        else:
            Logger.log('e', 'Single instance server was not created.')

    def _onClientConnected(self) -> None:
        if False:
            i = 10
            return i + 15
        Logger.log('i', 'New connection received on our single-instance server')
        connection = None
        if self._single_instance_server:
            connection = self._single_instance_server.nextPendingConnection()
        if connection is not None:
            connection.readyRead.connect(lambda c=connection: self.__readCommands(c))

    def __readCommands(self, connection: QLocalSocket) -> None:
        if False:
            while True:
                i = 10
        line = connection.readLine()
        while len(line) != 0:
            try:
                payload = json.loads(str(line, encoding='ascii').strip())
                command = payload['command']
                if command == 'clear-all':
                    self._application.callLater(lambda : self._application.deleteAll())
                elif command == 'open':
                    self._application.callLater(lambda f=payload['filePath']: self._application._openFile(f))
                elif command == 'focus':
                    main_window = self._application.getMainWindow()
                    if main_window is not None:
                        self._application.callLater(lambda : main_window.alert(0))
                elif command == 'close-connection':
                    connection.close()
                else:
                    Logger.log('w', 'Received an unrecognized command ' + str(command))
            except json.decoder.JSONDecodeError as ex:
                Logger.log('w', "Unable to parse JSON command '%s': %s", line, repr(ex))
            line = connection.readLine()