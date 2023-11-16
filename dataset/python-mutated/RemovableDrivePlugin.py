import threading
import time
from UM.Logger import Logger
from UM.OutputDevice.OutputDevicePlugin import OutputDevicePlugin
from . import RemovableDriveOutputDevice
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class RemovableDrivePlugin(OutputDevicePlugin):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self._update_thread = threading.Thread(target=self._updateThread)
        self._update_thread.daemon = True
        self._check_updates = True
        self._drives = {}

    def start(self):
        if False:
            print('Hello World!')
        self._update_thread.start()

    def stop(self):
        if False:
            i = 10
            return i + 15
        self._check_updates = False
        self._update_thread.join()
        self._addRemoveDrives({})

    def checkRemovableDrives(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def ejectDevice(self, device):
        if False:
            while True:
                i = 10
        try:
            Logger.log('i', 'Attempting to eject the device')
            result = self.performEjectDevice(device)
        except Exception as e:
            Logger.log('e', 'Ejection failed due to: %s' % str(e))
            result = False
        if result:
            Logger.log('i', 'Successfully ejected the device')
        return result

    def performEjectDevice(self, device):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _updateThread(self):
        if False:
            while True:
                i = 10
        while self._check_updates:
            result = self.checkRemovableDrives()
            self._addRemoveDrives(result)
            time.sleep(5)

    def _addRemoveDrives(self, drives):
        if False:
            for i in range(10):
                print('nop')
        for (key, value) in drives.items():
            if key not in self._drives:
                self.getOutputDeviceManager().addOutputDevice(RemovableDriveOutputDevice.RemovableDriveOutputDevice(key, value))
                continue
            if self._drives[key] != value:
                self.getOutputDeviceManager().removeOutputDevice(key)
                self.getOutputDeviceManager().addOutputDevice(RemovableDriveOutputDevice.RemovableDriveOutputDevice(key, value))
        for key in self._drives.keys():
            if key not in drives:
                self.getOutputDeviceManager().removeOutputDevice(key)
        self._drives = drives