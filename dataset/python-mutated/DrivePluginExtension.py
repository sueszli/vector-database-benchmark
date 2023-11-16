import os
from datetime import datetime
from typing import Any, cast, Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSlot, pyqtProperty, pyqtSignal
from UM.Extension import Extension
from UM.Logger import Logger
from UM.Message import Message
from cura.CuraApplication import CuraApplication
from .Settings import Settings
from .DriveApiService import DriveApiService
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class DrivePluginExtension(QObject, Extension):
    backupsChanged = pyqtSignal()
    restoringStateChanged = pyqtSignal()
    creatingStateChanged = pyqtSignal()
    preferencesChanged = pyqtSignal()
    backupIdBeingRestoredChanged = pyqtSignal(arguments=['backup_id_being_restored'])
    DATE_FORMAT = '%d/%m/%Y %H:%M:%S'

    def __init__(self) -> None:
        if False:
            return 10
        QObject.__init__(self, None)
        Extension.__init__(self)
        self._drive_window = None
        self._backups = []
        self._is_restoring_backup = False
        self._is_creating_backup = False
        self._backup_id_being_restored = ''
        preferences = CuraApplication.getInstance().getPreferences()
        self._drive_api_service = DriveApiService()
        CuraApplication.getInstance().getCuraAPI().account.loginStateChanged.connect(self._onLoginStateChanged)
        CuraApplication.getInstance().applicationShuttingDown.connect(self._onApplicationShuttingDown)
        self._drive_api_service.restoringStateChanged.connect(self._onRestoringStateChanged)
        self._drive_api_service.creatingStateChanged.connect(self._onCreatingStateChanged)
        preferences.addPreference(Settings.AUTO_BACKUP_ENABLED_PREFERENCE_KEY, False)
        preferences.addPreference(Settings.AUTO_BACKUP_LAST_DATE_PREFERENCE_KEY, datetime.now().strftime(self.DATE_FORMAT))
        self.addMenuItem(catalog.i18nc('@item:inmenu', 'Manage backups'), self.showDriveWindow)
        CuraApplication.getInstance().engineCreatedSignal.connect(self._autoBackup)

    def showDriveWindow(self) -> None:
        if False:
            i = 10
            return i + 15
        if not self._drive_window:
            plugin_dir_path = cast(str, CuraApplication.getInstance().getPluginRegistry().getPluginPath(self.getPluginId()))
            path = os.path.join(plugin_dir_path, 'src', 'qml', 'main.qml')
            self._drive_window = CuraApplication.getInstance().createQmlComponent(path, {'CuraDrive': self})
        self.refreshBackups()
        if self._drive_window:
            self._drive_window.show()

    def _onApplicationShuttingDown(self):
        if False:
            i = 10
            return i + 15
        if self._drive_window:
            self._drive_window.hide()

    def _autoBackup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        preferences = CuraApplication.getInstance().getPreferences()
        if preferences.getValue(Settings.AUTO_BACKUP_ENABLED_PREFERENCE_KEY) and self._isLastBackupTooLongAgo():
            self.createBackup()

    def _isLastBackupTooLongAgo(self) -> bool:
        if False:
            while True:
                i = 10
        current_date = datetime.now()
        last_backup_date = self._getLastBackupDate()
        date_diff = current_date - last_backup_date
        return date_diff.days > 1

    def _getLastBackupDate(self) -> 'datetime':
        if False:
            while True:
                i = 10
        preferences = CuraApplication.getInstance().getPreferences()
        last_backup_date = preferences.getValue(Settings.AUTO_BACKUP_LAST_DATE_PREFERENCE_KEY)
        return datetime.strptime(last_backup_date, self.DATE_FORMAT)

    def _storeBackupDate(self) -> None:
        if False:
            return 10
        backup_date = datetime.now().strftime(self.DATE_FORMAT)
        preferences = CuraApplication.getInstance().getPreferences()
        preferences.setValue(Settings.AUTO_BACKUP_LAST_DATE_PREFERENCE_KEY, backup_date)

    def _onLoginStateChanged(self, logged_in: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        if logged_in:
            self.refreshBackups()

    def _onRestoringStateChanged(self, is_restoring: bool=False, error_message: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        self._is_restoring_backup = is_restoring
        self.restoringStateChanged.emit()
        if error_message:
            self.backupIdBeingRestored = ''
            Message(error_message, title=catalog.i18nc('@info:title', 'Backup'), message_type=Message.MessageType.ERROR).show()

    def _onCreatingStateChanged(self, is_creating: bool=False, error_message: str=None) -> None:
        if False:
            return 10
        self._is_creating_backup = is_creating
        self.creatingStateChanged.emit()
        if error_message:
            Message(error_message, title=catalog.i18nc('@info:title', 'Backup'), message_type=Message.MessageType.ERROR).show()
        else:
            self._storeBackupDate()
        if not is_creating and (not error_message):
            self.refreshBackups()

    @pyqtSlot(bool, name='toggleAutoBackup')
    def toggleAutoBackup(self, enabled: bool) -> None:
        if False:
            print('Hello World!')
        preferences = CuraApplication.getInstance().getPreferences()
        preferences.setValue(Settings.AUTO_BACKUP_ENABLED_PREFERENCE_KEY, enabled)

    @pyqtProperty(bool, notify=preferencesChanged)
    def autoBackupEnabled(self) -> bool:
        if False:
            return 10
        preferences = CuraApplication.getInstance().getPreferences()
        return bool(preferences.getValue(Settings.AUTO_BACKUP_ENABLED_PREFERENCE_KEY))

    @pyqtProperty('QVariantList', notify=backupsChanged)
    def backups(self) -> List[Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        return self._backups

    @pyqtSlot(name='refreshBackups')
    def refreshBackups(self) -> None:
        if False:
            while True:
                i = 10
        self._drive_api_service.getBackups(self._backupsChangedCallback)

    def _backupsChangedCallback(self, backups: List[Dict[str, Any]]) -> None:
        if False:
            while True:
                i = 10
        self._backups = backups
        self.backupsChanged.emit()

    @pyqtProperty(bool, notify=restoringStateChanged)
    def isRestoringBackup(self) -> bool:
        if False:
            return 10
        return self._is_restoring_backup

    @pyqtProperty(bool, notify=creatingStateChanged)
    def isCreatingBackup(self) -> bool:
        if False:
            return 10
        return self._is_creating_backup

    @pyqtSlot(str, name='restoreBackup')
    def restoreBackup(self, backup_id: str) -> None:
        if False:
            while True:
                i = 10
        for backup in self._backups:
            if backup.get('backup_id') == backup_id:
                self._drive_api_service.restoreBackup(backup)
                self.setBackupIdBeingRestored(backup_id)
                return
        Logger.log('w', 'Unable to find backup with the ID %s', backup_id)

    @pyqtSlot(name='createBackup')
    def createBackup(self) -> None:
        if False:
            i = 10
            return i + 15
        self._drive_api_service.createBackup()

    @pyqtSlot(str, name='deleteBackup')
    def deleteBackup(self, backup_id: str) -> None:
        if False:
            i = 10
            return i + 15
        self._drive_api_service.deleteBackup(backup_id, self._backupDeletedCallback)

    def _backupDeletedCallback(self, success: bool):
        if False:
            for i in range(10):
                print('nop')
        if success:
            self.refreshBackups()

    def setBackupIdBeingRestored(self, backup_id_being_restored: str) -> None:
        if False:
            while True:
                i = 10
        if backup_id_being_restored != self._backup_id_being_restored:
            self._backup_id_being_restored = backup_id_being_restored
            self.backupIdBeingRestoredChanged.emit()

    @pyqtProperty(str, fset=setBackupIdBeingRestored, notify=backupIdBeingRestoredChanged)
    def backupIdBeingRestored(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._backup_id_being_restored