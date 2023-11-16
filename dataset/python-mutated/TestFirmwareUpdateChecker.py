import pytest
from unittest.mock import MagicMock
from UM.Version import Version
import FirmwareUpdateChecker
json_data = {'ned': {'id': 1, 'name': 'ned', 'check_urls': [''], 'update_url': 'https://ultimaker.com/en/resources/20500-upgrade-firmware', 'version_parser': 'default'}, 'olivia': {'id': 3, 'name': 'olivia', 'check_urls': [''], 'update_url': 'https://ultimaker.com/en/resources/20500-upgrade-firmware', 'version_parser': 'default'}, 'emmerson': {'id': 5, 'name': 'emmerson', 'check_urls': [''], 'update_url': 'https://ultimaker.com/en/resources/20500-upgrade-firmware', 'version_parser': 'default'}}

@pytest.mark.parametrize('name, id', [('ned', 1), ('olivia', 3), ('emmerson', 5)])
def test_FirmwareUpdateCheckerLookup(id, name):
    if False:
        print('Hello World!')
    lookup = FirmwareUpdateChecker.FirmwareUpdateCheckerLookup.FirmwareUpdateCheckerLookup(name, json_data.get(name))
    assert lookup.getMachineName() == name
    assert lookup.getMachineId() == id
    assert len(lookup.getCheckUrls()) >= 1
    assert lookup.getRedirectUserUrl() is not None

@pytest.mark.parametrize('name, version', [('ned', Version('5.1.2.3')), ('olivia', Version('4.3.2.1')), ('emmerson', Version('6.7.8.1'))])
def test_FirmwareUpdateCheckerJob_getCurrentVersion(name, version):
    if False:
        print('Hello World!')
    machine_data = json_data.get(name)
    job = FirmwareUpdateChecker.FirmwareUpdateCheckerJob.FirmwareUpdateCheckerJob(False, name, machine_data, MagicMock)
    job.getUrlResponse = MagicMock(return_value=str(version))
    assert job.getCurrentVersion() == version