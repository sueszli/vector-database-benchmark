import socket
import os
import subprocess
from typing import Optional, List
from UM.Logger import Logger
from UM.Message import Message
from UM.Settings.AdditionalSettingDefinitionsAppender import AdditionalSettingDefinitionsAppender
from UM.PluginObject import PluginObject
from UM.i18n import i18nCatalog
from UM.Platform import Platform
from UM.Resources import Resources

class BackendPlugin(AdditionalSettingDefinitionsAppender, PluginObject):
    catalog = i18nCatalog('cura')
    settings_catalog = i18nCatalog('fdmprinter.def.json')

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(self.settings_catalog)
        self.__port: int = 0
        self._plugin_address: str = '127.0.0.1'
        self._plugin_command: Optional[List[str]] = None
        self._process = None
        self._is_running = False
        self._supported_slots: List[int] = []
        self._use_plugin = True

    def usePlugin(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._use_plugin

    def getSupportedSlots(self) -> List[int]:
        if False:
            i = 10
            return i + 15
        return self._supported_slots

    def isRunning(self):
        if False:
            return 10
        return self._is_running

    def setPort(self, port: int) -> None:
        if False:
            print('Hello World!')
        self.__port = port

    def getPort(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.__port

    def getAddress(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._plugin_address

    def setAvailablePort(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the port to a random available port.\n        '
        sock = socket.socket()
        sock.bind((self.getAddress(), 0))
        port = sock.getsockname()[1]
        self.setPort(port)

    def _validatePluginCommand(self) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate the plugin command and add the port parameter if it is missing.\n\n        :return: A list of strings containing the validated plugin command.\n        '
        if not self._plugin_command or '--port' in self._plugin_command:
            return self._plugin_command or []
        return self._plugin_command + ['--address', self.getAddress(), '--port', str(self.__port)]

    def start(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Starts the backend_plugin process.\n\n        :return: True if the plugin process started successfully, False otherwise.\n        '
        if not self.usePlugin():
            return False
        Logger.info(f'Starting backend_plugin [{self._plugin_id}] with command: {self._validatePluginCommand()}')
        plugin_log_path = os.path.join(Resources.getDataStoragePath(), f'{self.getPluginId()}.log')
        if os.path.exists(plugin_log_path):
            try:
                os.remove(plugin_log_path)
            except:
                pass
        Logger.info(f'Logging plugin output to: {plugin_log_path}')
        try:
            with open(plugin_log_path, 'a') as f:
                popen_kwargs = {'stdin': None, 'stdout': f, 'stderr': subprocess.STDOUT}
                if Platform.isWindows():
                    popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                self._process = subprocess.Popen(self._validatePluginCommand(), **popen_kwargs)
            self._is_running = True
            return True
        except PermissionError:
            Logger.log('e', f"Couldn't start EnginePlugin: {self._plugin_id} No permission to execute process.")
            self._showMessage(self.catalog.i18nc('@info:plugin_failed', f"Couldn't start EnginePlugin: {self._plugin_id}\nNo permission to execute process."), message_type=Message.MessageType.ERROR)
        except FileNotFoundError:
            Logger.logException('e', f'Unable to find local EnginePlugin server executable for: {self._plugin_id}')
            self._showMessage(self.catalog.i18nc('@info:plugin_failed', f'Unable to find local EnginePlugin server executable for: {self._plugin_id}'), message_type=Message.MessageType.ERROR)
        except BlockingIOError:
            Logger.logException('e', f"Couldn't start EnginePlugin: {self._plugin_id} Resource is temporarily unavailable")
            self._showMessage(self.catalog.i18nc('@info:plugin_failed', f"Couldn't start EnginePlugin: {self._plugin_id}\nResource is temporarily unavailable"), message_type=Message.MessageType.ERROR)
        except OSError as e:
            Logger.logException('e', f"Couldn't start EnginePlugin {self._plugin_id} Operating system is blocking it (antivirus?)")
            self._showMessage(self.catalog.i18nc('@info:plugin_failed', f"Couldn't start EnginePlugin: {self._plugin_id}\nOperating system is blocking it (antivirus?)"), message_type=Message.MessageType.ERROR)
        return False

    def stop(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not self._process:
            self._is_running = False
            return True
        try:
            self._process.terminate()
            return_code = self._process.wait()
            self._is_running = False
            Logger.log('d', f'EnginePlugin: {self._plugin_id} was killed. Received return code {return_code}')
            return True
        except PermissionError:
            Logger.log('e', f'Unable to kill running EnginePlugin: {self._plugin_id} Access is denied.')
            self._showMessage(self.catalog.i18nc('@info:plugin_failed', f'Unable to kill running EnginePlugin: {self._plugin_id}\nAccess is denied.'), message_type=Message.MessageType.ERROR)
            return False

    def _showMessage(self, message: str, message_type: Message.MessageType=Message.MessageType.ERROR) -> None:
        if False:
            for i in range(10):
                print('nop')
        Message(message, title=self.catalog.i18nc('@info:title', 'EnginePlugin'), message_type=message_type).show()