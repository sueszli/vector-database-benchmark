"""Mock devices object to test Insteon."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pyinsteon.address import Address
from pyinsteon.constants import ALDBStatus, ResponseStatus
from pyinsteon.device_types.ipdb import AccessControl_Morningstar, DimmableLightingControl_KeypadLinc_8, GeneralController_RemoteLinc, Hub, SensorsActuators_IOLink, SwitchedLightingControl_SwitchLinc02
from pyinsteon.managers.saved_devices_manager import dict_to_aldb_record
from pyinsteon.topics import DEVICE_LIST_CHANGED
from pyinsteon.utils import subscribe_topic

class MockSwitchLinc(SwitchedLightingControl_SwitchLinc02):
    """Mock SwitchLinc device."""

    @property
    def operating_flags(self):
        if False:
            for i in range(10):
                print('nop')
        'Return no operating flags to force properties to be checked.'
        return {}

class MockDevices:
    """Mock devices class."""

    def __init__(self, connected=True):
        if False:
            i = 10
            return i + 15
        'Init the MockDevices class.'
        self._devices = {}
        self.modem = None
        self._connected = connected
        self.async_save = AsyncMock()
        self.add_x10_device = MagicMock()
        self.async_read_config = AsyncMock()
        self.set_id = MagicMock()
        self.async_add_device_called_with = {}
        self.async_cancel_all_linking = AsyncMock()

    def __getitem__(self, address):
        if False:
            for i in range(10):
                print('nop')
        'Return a a device from the device address.'
        return self._devices.get(Address(address))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return an iterator of device addresses.'
        yield from self._devices

    def __len__(self):
        if False:
            print('Hello World!')
        'Return the number of devices.'
        return len(self._devices)

    def get(self, address):
        if False:
            return 10
        'Return a device from an address or None if not found.'
        return self._devices.get(Address(address))

    async def async_load(self, *args, **kwargs):
        """Load the mock devices."""
        if self._connected and (not self._devices):
            addr0 = Address('AA.AA.AA')
            addr1 = Address('11.11.11')
            addr2 = Address('22.22.22')
            addr3 = Address('33.33.33')
            addr4 = Address('44.44.44')
            addr5 = Address('55.55.55')
            self._devices[addr0] = Hub(addr0, 3, 0, 0, 'Hub AA.AA.AA', '0')
            self._devices[addr1] = MockSwitchLinc(addr1, 2, 0, 0, 'Device 11.11.11', '1')
            self._devices[addr2] = GeneralController_RemoteLinc(addr2, 0, 0, 0, 'Device 22.22.22', '2')
            self._devices[addr3] = DimmableLightingControl_KeypadLinc_8(addr3, 2, 0, 0, 'Device 33.33.33', '3')
            self._devices[addr4] = SensorsActuators_IOLink(addr4, 7, 0, 0, 'Device 44.44.44', '4')
            self._devices[addr5] = AccessControl_Morningstar(addr5, 15, 10, 0, 'Device 55.55.55', '5')
            for device in [self._devices[addr] for addr in [addr1, addr2, addr3, addr4, addr5]]:
                device.async_read_config = AsyncMock()
                device.aldb.async_write = AsyncMock()
                device.aldb.async_load = AsyncMock()
                device.async_add_default_links = AsyncMock()
                device.async_read_op_flags = AsyncMock(return_value=ResponseStatus.SUCCESS)
                device.async_read_ext_properties = AsyncMock(return_value=ResponseStatus.SUCCESS)
                device.async_write_op_flags = AsyncMock(return_value=ResponseStatus.SUCCESS)
                device.async_write_ext_properties = AsyncMock(return_value=ResponseStatus.SUCCESS)
            for device in [self._devices[addr] for addr in [addr2, addr3, addr4, addr5]]:
                device.async_status = AsyncMock()
            self._devices[addr1].async_status = AsyncMock(side_effect=AttributeError)
            self._devices[addr0].aldb.async_load = AsyncMock()
            self._devices[addr2].async_read_op_flags = AsyncMock(return_value=ResponseStatus.FAILURE)
            self._devices[addr2].async_read_ext_properties = AsyncMock(return_value=ResponseStatus.FAILURE)
            self._devices[addr2].async_write_op_flags = AsyncMock(return_value=ResponseStatus.FAILURE)
            self._devices[addr2].async_write_ext_properties = AsyncMock(return_value=ResponseStatus.FAILURE)
            self._devices[addr5].async_lock = AsyncMock(return_value=ResponseStatus.SUCCESS)
            self._devices[addr5].async_unlock = AsyncMock(return_value=ResponseStatus.SUCCESS)
            self.modem = self._devices[addr0]
            self.modem.async_read_config = AsyncMock()

    def fill_aldb(self, address, records):
        if False:
            for i in range(10):
                print('nop')
        'Fill the All-Link Database for a device.'
        device = self._devices[Address(address)]
        aldb_records = dict_to_aldb_record(records)
        device.aldb.load_saved_records(ALDBStatus.LOADED, aldb_records)

    def fill_properties(self, address, props_dict):
        if False:
            while True:
                i = 10
        'Fill the operating flags and extended properties of a device.'
        device = self._devices[Address(address)]
        operating_flags = props_dict.get('operating_flags', {})
        properties = props_dict.get('properties', {})
        with patch('pyinsteon.subscriber_base.publish_topic', MagicMock()):
            for flag in operating_flags:
                value = operating_flags[flag]
                if device.operating_flags.get(flag):
                    device.operating_flags[flag].set_value(value)
            for flag in properties:
                value = properties[flag]
                if device.properties.get(flag):
                    device.properties[flag].set_value(value)

    async def async_add_device(self, address=None, multiple=False):
        """Mock the async_add_device method."""
        self.async_add_device_called_with = {'address': address, 'multiple': multiple}
        if multiple:
            yield 'aa.bb.cc'
            await asyncio.sleep(0.01)
            yield 'bb.cc.dd'
        if address:
            yield address
        await asyncio.sleep(0.01)

    def subscribe(self, listener, force_strong_ref=False):
        if False:
            for i in range(10):
                print('nop')
        'Mock the subscribe function.'
        subscribe_topic(listener, DEVICE_LIST_CHANGED)