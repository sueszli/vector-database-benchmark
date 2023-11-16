"""
    :codeauthor: Pablo Suárez Hdez. <psuarezhernandez@suse.de>

    Test cases for salt.modules.udev
"""
import pytest
import salt.modules.udev as udev
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {udev: {}}

def test_info():
    if False:
        while True:
            i = 10
    '\n    Test if it returns the info of udev-created node in a dict\n    '
    cmd_out = {'retcode': 0, 'stdout': 'P: /devices/virtual/vc/vcsa7\nN: vcsa7\nE: DEVNAME=/dev/vcsa7\nE: DEVPATH=/devices/virtual/vc/vcsa7\nE: MAJOR=7\nE: MINOR=135\nE: SUBSYSTEM=vc\n\n', 'stderr': ''}
    ret = {'E': {'DEVNAME': '/dev/vcsa7', 'DEVPATH': '/devices/virtual/vc/vcsa7', 'MAJOR': 7, 'MINOR': 135, 'SUBSYSTEM': 'vc'}, 'N': 'vcsa7', 'P': '/devices/virtual/vc/vcsa7'}
    mock = MagicMock(return_value=cmd_out)
    with patch.dict(udev.__salt__, {'cmd.run_all': mock}):
        data = udev.info('/dev/vcsa7')
        assert ret['P'] == data['P']
        assert ret.get('N') == data.get('N')
        for (key, value) in data['E'].items():
            assert ret['E'][key] == value

def test_exportdb():
    if False:
        return 10
    '\n    Test if it returns the all the udev database into a dict\n    '
    udev_data = '\nP: /devices/LNXSYSTM:00/LNXPWRBN:00\nE: DEVPATH=/devices/LNXSYSTM:00/LNXPWRBN:00\nE: DRIVER=button\nE: MODALIAS=acpi:LNXPWRBN:\nE: SUBSYSTEM=acpi\n\nP: /devices/LNXSYSTM:00/LNXPWRBN:00/input/input2\nE: DEVPATH=/devices/LNXSYSTM:00/LNXPWRBN:00/input/input2\nE: EV=3\nE: ID_FOR_SEAT=input-acpi-LNXPWRBN_00\nE: ID_INPUT=1\nE: ID_INPUT_KEY=1\nE: ID_PATH=acpi-LNXPWRBN:00\nE: ID_PATH_TAG=acpi-LNXPWRBN_00\nE: KEY=10000000000000 0\nE: MODALIAS=input:b0019v0000p0001e0000-e0,1,k74,ramlsfw\nE: NAME="Power Button"\nE: PHYS="LNXPWRBN/button/input0"\nE: PRODUCT=19/0/1/0\nE: PROP=0\nE: SUBSYSTEM=input\nE: TAGS=:seat:\nE: USEC_INITIALIZED=2010022\n\nP: /devices/LNXSYSTM:00/LNXPWRBN:00/input/input2/event2\nN: input/event2\nE: BACKSPACE=guess\nE: DEVNAME=/dev/input/event2\nE: DEVPATH=/devices/LNXSYSTM:00/LNXPWRBN:00/input/input2/event2\nE: ID_INPUT=1\nE: ID_INPUT_KEY=1\nE: ID_PATH=acpi-LNXPWRBN:00\nE: ID_PATH_TAG=acpi-LNXPWRBN_00\nE: MAJOR=13\nE: MINOR=66\nE: SUBSYSTEM=input\nE: TAGS=:power-switch:\nE: USEC_INITIALIZED=2076101\nE: XKBLAYOUT=us\nE: XKBMODEL=pc105\n'
    out = [{'P': '/devices/LNXSYSTM:00/LNXPWRBN:00', 'E': {'MODALIAS': 'acpi:LNXPWRBN:', 'SUBSYSTEM': 'acpi', 'DRIVER': 'button', 'DEVPATH': '/devices/LNXSYSTM:00/LNXPWRBN:00'}}, {'P': '/devices/LNXSYSTM:00/LNXPWRBN:00/input/input2', 'E': {'SUBSYSTEM': 'input', 'PRODUCT': '19/0/1/0', 'PHYS': '"LNXPWRBN/button/input0"', 'NAME': '"Power Button"', 'ID_INPUT': 1, 'DEVPATH': '/devices/LNXSYSTM:00/LNXPWRBN:00/input/input2', 'MODALIAS': 'input:b0019v0000p0001e0000-e0,1,k74,ramlsfw', 'ID_PATH_TAG': 'acpi-LNXPWRBN_00', 'TAGS': ':seat:', 'PROP': 0, 'ID_FOR_SEAT': 'input-acpi-LNXPWRBN_00', 'KEY': '10000000000000 0', 'USEC_INITIALIZED': 2010022, 'ID_PATH': 'acpi-LNXPWRBN:00', 'EV': 3, 'ID_INPUT_KEY': 1}}, {'P': '/devices/LNXSYSTM:00/LNXPWRBN:00/input/input2/event2', 'E': {'SUBSYSTEM': 'input', 'XKBLAYOUT': 'us', 'MAJOR': 13, 'ID_INPUT': 1, 'DEVPATH': '/devices/LNXSYSTM:00/LNXPWRBN:00/input/input2/event2', 'ID_PATH_TAG': 'acpi-LNXPWRBN_00', 'DEVNAME': '/dev/input/event2', 'TAGS': ':power-switch:', 'BACKSPACE': 'guess', 'MINOR': 66, 'USEC_INITIALIZED': 2076101, 'ID_PATH': 'acpi-LNXPWRBN:00', 'XKBMODEL': 'pc105', 'ID_INPUT_KEY': 1}, 'N': 'input/event2'}]
    mock = MagicMock(return_value={'retcode': 0, 'stdout': udev_data})
    with patch.dict(udev.__salt__, {'cmd.run_all': mock}):
        data = udev.exportdb()
        assert data == [x for x in data if x]
        for (d_idx, d_section) in enumerate(data):
            assert out[d_idx]['P'] == d_section['P']
            assert out[d_idx].get('N') == d_section.get('N')
            for (key, value) in d_section['E'].items():
                assert out[d_idx]['E'][key] == value

def test_normalize_info():
    if False:
        while True:
            i = 10
    '\n    Test if udevdb._normalize_info does not returns nested lists that contains only one item.\n\n    :return:\n    '
    data = {'key': ['value', 'here'], 'foo': ['bar'], 'some': 'data'}
    assert udev._normalize_info(data) == {'foo': 'bar', 'some': 'data', 'key': ['value', 'here']}