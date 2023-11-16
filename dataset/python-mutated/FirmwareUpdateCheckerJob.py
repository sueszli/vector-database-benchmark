from UM.Application import Application
from UM.Message import Message
from UM.Logger import Logger
from UM.Job import Job
from UM.Version import Version
import urllib.request
from urllib.error import URLError
from typing import Dict
import ssl
import certifi
from .FirmwareUpdateCheckerLookup import FirmwareUpdateCheckerLookup, getSettingsKeyForMachine
from .FirmwareUpdateCheckerMessage import FirmwareUpdateCheckerMessage
from UM.i18n import i18nCatalog
i18n_catalog = i18nCatalog('cura')

class FirmwareUpdateCheckerJob(Job):
    """This job checks if there is an update available on the provided URL."""
    STRING_ZERO_VERSION = '0.0.0'
    STRING_EPSILON_VERSION = '0.0.1'
    ZERO_VERSION = Version(STRING_ZERO_VERSION)
    EPSILON_VERSION = Version(STRING_EPSILON_VERSION)

    def __init__(self, silent, machine_name, metadata, callback) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.silent = silent
        self._callback = callback
        self._machine_name = machine_name
        self._metadata = metadata
        self._lookups = FirmwareUpdateCheckerLookup(self._machine_name, self._metadata)
        self._headers = {}

    def getUrlResponse(self, url: str) -> str:
        if False:
            return 10
        result = self.STRING_ZERO_VERSION
        try:
            context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLSv1_2)
            context.verify_mode = ssl.CERT_REQUIRED
            context.load_verify_locations(cafile=certifi.where())
            request = urllib.request.Request(url, headers=self._headers)
            response = urllib.request.urlopen(request, context=context)
            result = response.read().decode('utf-8')
        except URLError:
            Logger.log('w', "Could not reach '{0}', if this URL is old, consider removal.".format(url))
        return result

    def parseVersionResponse(self, response: str) -> Version:
        if False:
            i = 10
            return i + 15
        raw_str = response.split('\n', 1)[0].rstrip()
        return Version(raw_str)

    def getCurrentVersion(self) -> Version:
        if False:
            return 10
        max_version = self.ZERO_VERSION
        if self._lookups is None:
            return max_version
        machine_urls = self._lookups.getCheckUrls()
        if machine_urls is not None:
            for url in machine_urls:
                version = self.parseVersionResponse(self.getUrlResponse(url))
                if version > max_version:
                    max_version = version
        if max_version < self.EPSILON_VERSION:
            Logger.log('w', 'MachineID {0} not handled!'.format(self._lookups.getMachineName()))
        return max_version

    def run(self):
        if False:
            while True:
                i = 10
        try:
            Application.getInstance().getPreferences().addPreference(getSettingsKeyForMachine(self._lookups.getMachineId()), '')
            application_name = Application.getInstance().getApplicationName()
            application_version = Application.getInstance().getVersion()
            self._headers = {'User-Agent': '%s - %s' % (application_name, application_version)}
            machine_id = self._lookups.getMachineId()
            if machine_id is not None:
                Logger.log('i', 'You have a(n) {0} in the printer list. Do firmware-check.'.format(self._machine_name))
                current_version = self.getCurrentVersion()
                if current_version == self.ZERO_VERSION:
                    return
                setting_key_str = getSettingsKeyForMachine(machine_id)
                checked_version = Version(Application.getInstance().getPreferences().getValue(setting_key_str))
                Application.getInstance().getPreferences().setValue(setting_key_str, current_version)
                Logger.log('i', 'Reading firmware version of %s: checked = %s - latest = %s', self._machine_name, checked_version, current_version)
                if checked_version != '' and checked_version != current_version:
                    Logger.log('i', 'Showing firmware update message for new version: {version}'.format(version=current_version))
                    message = FirmwareUpdateCheckerMessage(machine_id, self._machine_name, current_version, self._lookups.getRedirectUserUrl())
                    message.actionTriggered.connect(self._callback)
                    message.show()
            else:
                Logger.log('i', 'No machine with name {0} in list of firmware to check.'.format(self._machine_name))
        except Exception as e:
            Logger.logException('w', 'Failed to check for new version: %s', e)
            if not self.silent:
                Message(i18n_catalog.i18nc('@info', 'Could not access update information.')).show()
            return