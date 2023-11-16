from typing import List, Optional
from UM.i18n import i18nCatalog
i18n_catalog = i18nCatalog('cura')

def getSettingsKeyForMachine(machine_id: int) -> str:
    if False:
        return 10
    return 'info/latest_checked_firmware_for_{0}'.format(machine_id)

class FirmwareUpdateCheckerLookup:

    def __init__(self, machine_name, machine_json) -> None:
        if False:
            i = 10
            return i + 15
        self._machine_id = machine_json.get('id')
        self._machine_name = machine_name.lower()
        self._check_urls = []
        for check_url in machine_json.get('check_urls', []):
            self._check_urls.append(check_url)
        self._redirect_user = machine_json.get('update_url')

    def getMachineId(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        return self._machine_id

    def getMachineName(self) -> Optional[int]:
        if False:
            return 10
        return self._machine_name

    def getCheckUrls(self) -> Optional[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        return self._check_urls

    def getRedirectUserUrl(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._redirect_user