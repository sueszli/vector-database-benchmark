import asyncio
import os
import shutil
import signal
import subprocess
import time
from enum import Enum
from threading import Thread
import pytest
from dbus_next._private.address import get_session_bus_address
from dbus_next.aio import MessageBus
from dbus_next.constants import PropertyAccess
from dbus_next.service import ServiceInterface, dbus_property, method
from libqtile.bar import Bar
from libqtile.config import Screen
from libqtile.widget.bluetooth import BLUEZ_ADAPTER, BLUEZ_BATTERY, BLUEZ_DEVICE, Bluetooth
from test.conftest import BareConfig
from test.helpers import Retry
ADAPTER_PATH = '/org/bluez/hci0'
ADAPTER_NAME = 'qtile_bluez'
BLUEZ_SERVICE = 'test.qtile.bluez'

class DeviceState(Enum):
    UNPAIRED = 1
    PAIRED = 2
    CONNECTED = 3

class Device(ServiceInterface):

    def __init__(self, *args, alias, state, adapter, address, **kwargs):
        if False:
            while True:
                i = 10
        ServiceInterface.__init__(self, *args, **kwargs)
        self._state = state
        self._name = alias
        self._adapter = adapter
        self._address = ':'.join([address] * 8)

    @method()
    def Pair(self):
        if False:
            while True:
                i = 10
        self._state = DeviceState.PAIRED
        self.emit_properties_changed({'Paired': True, 'Connected': False})

    @method()
    def Connect(self):
        if False:
            while True:
                i = 10
        self._state = DeviceState.CONNECTED
        self.emit_properties_changed({'Paired': True, 'Connected': True})

    @method()
    def Disconnect(self):
        if False:
            print('Hello World!')
        self._state = DeviceState.PAIRED
        self.emit_properties_changed({'Paired': True, 'Connected': False})

    @dbus_property(access=PropertyAccess.READ)
    def Name(self) -> 's':
        if False:
            print('Hello World!')
        return self._name

    @dbus_property(access=PropertyAccess.READ)
    def Address(self) -> 's':
        if False:
            for i in range(10):
                print('nop')
        return self._address

    @dbus_property(access=PropertyAccess.READ)
    def Adapter(self) -> 's':
        if False:
            print('Hello World!')
        return self._adapter

    @dbus_property(access=PropertyAccess.READ)
    def Connected(self) -> 'b':
        if False:
            return 10
        return self._state == DeviceState.CONNECTED

    @dbus_property(access=PropertyAccess.READ)
    def Paired(self) -> 'b':
        if False:
            while True:
                i = 10
        return self._state != DeviceState.UNPAIRED

class Adapter(ServiceInterface):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ServiceInterface.__init__(self, *args, **kwargs)
        self._name = ADAPTER_NAME
        self._powered = True
        self._discovering = False

    @dbus_property(access=PropertyAccess.READ)
    def Name(self) -> 's':
        if False:
            while True:
                i = 10
        return self._name

    @dbus_property()
    def Powered(self) -> 'b':
        if False:
            for i in range(10):
                print('nop')
        return self._powered

    @Powered.setter
    def Powered_setter(self, state: 'b'):
        if False:
            return 10
        self._powered = state
        self.emit_properties_changed({'Powered': state})

    @dbus_property(access=PropertyAccess.READ)
    def Discovering(self) -> 'b':
        if False:
            i = 10
            return i + 15
        return self._discovering

    @method()
    def StartDiscovery(self):
        if False:
            return 10
        self._discovering = True
        self.emit_properties_changed({'Discovering': self._discovering})

    @method()
    def StopDiscovery(self):
        if False:
            return 10
        self._discovering = False
        self.emit_properties_changed({'Discovering': self._discovering})

class Battery(ServiceInterface):

    @dbus_property(PropertyAccess.READ)
    def Percentage(self) -> 'd':
        if False:
            while True:
                i = 10
        return 75

class Bluez(Thread):
    """Class that runs fake UPower interface in a thread."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        Thread.__init__(self, *args, **kwargs)

    async def start_server(self):
        """Connects to the bus and publishes 3 interfaces."""
        bus = await MessageBus().connect()
        root = ServiceInterface('org.qtile.root')
        bus.export('/', root)
        unpaired_device = Device(BLUEZ_DEVICE, alias='Earbuds', state=DeviceState.UNPAIRED, address='11', adapter=ADAPTER_PATH)
        paired_device = Device(BLUEZ_DEVICE, alias='Headphones', state=DeviceState.PAIRED, address='22', adapter=ADAPTER_PATH)
        connected_device = Device(BLUEZ_DEVICE, alias='Speaker', state=DeviceState.CONNECTED, address='33', adapter=ADAPTER_PATH)
        battery = Battery(BLUEZ_BATTERY)
        for d in [unpaired_device, paired_device, connected_device]:
            path = f"{ADAPTER_PATH}/dev_{d._address.replace(':', '_')}"
            bus.export(path, d)
            if d is connected_device:
                bus.export(path, battery)
        adapter = Adapter(BLUEZ_ADAPTER)
        bus.export(ADAPTER_PATH, adapter)
        await bus.request_name(BLUEZ_SERVICE)
        await asyncio.get_event_loop().create_future()

    def run(self):
        if False:
            while True:
                i = 10
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.start_server())

@pytest.fixture()
def dbus_thread(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Start a thread which publishes a fake bluez interface on dbus.'
    launcher = shutil.which('dbus-launch')
    if launcher is None:
        pytest.skip('dbus-launch must be installed')
    result = subprocess.run(launcher, capture_output=True)
    pid = None
    for line in result.stdout.decode().splitlines():
        (var, _, val) = line.partition('=')
        monkeypatch.setitem(os.environ, var, val)
        if var == 'DBUS_SESSION_BUS_PID':
            try:
                pid = int(val)
            except ValueError:
                pass
    t = Bluez()
    t.daemon = True
    t.start()
    time.sleep(1)
    yield
    if pid:
        os.kill(pid, signal.SIGTERM)

@pytest.fixture
def widget(monkeypatch):
    if False:
        return 10
    'Patch the widget to use the fake dbus service.'

    def force_session_bus(bus_type):
        if False:
            while True:
                i = 10
        return get_session_bus_address()
    monkeypatch.setattr('libqtile.widget.bluetooth.BLUEZ_SERVICE', BLUEZ_SERVICE)
    monkeypatch.setattr('dbus_next.message_bus.get_bus_address', force_session_bus)
    yield Bluetooth

@pytest.fixture
def bluetooth_manager(request, widget, dbus_thread, manager_nospawn):
    if False:
        i = 10
        return i + 15

    class BluetoothConfig(BareConfig):
        screens = [Screen(top=Bar([widget(**getattr(request, 'param', dict()))], 20))]
    manager_nospawn.start(BluetoothConfig)
    yield manager_nospawn

@Retry(ignore_exceptions=(AssertionError,))
def wait_for_text(widget, text):
    if False:
        return 10
    assert widget.info()['text'] == text

def test_defaults(bluetooth_manager):
    if False:
        return 10
    widget = bluetooth_manager.c.widget['bluetooth']

    def text():
        if False:
            return 10
        return widget.info()['text']

    def click():
        if False:
            while True:
                i = 10
        bluetooth_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    wait_for_text(widget, 'BT Speaker')
    widget.scroll_up()
    assert text() == f'Adapter: {ADAPTER_NAME} [*]'
    widget.scroll_up()
    assert text() == 'Device: Earbuds [?]'
    widget.scroll_up()
    assert text() == 'Device: Headphones [-]'
    widget.scroll_up()
    assert text() == 'Device: Speaker (75.0%) [*]'
    widget.scroll_up()
    assert text() == 'BT Speaker'
    widget.scroll_down()
    widget.scroll_down()
    assert text() == 'Device: Headphones [-]'

def test_device_actions(bluetooth_manager):
    if False:
        for i in range(10):
            print('nop')
    widget = bluetooth_manager.c.widget['bluetooth']

    def text():
        if False:
            return 10
        return widget.info()['text']

    def click():
        if False:
            return 10
        bluetooth_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    wait_for_text(widget, 'BT Speaker')
    widget.scroll_down()
    widget.scroll_down()
    wait_for_text(widget, 'Device: Headphones [-]')
    click()
    wait_for_text(widget, 'Device: Headphones [*]')
    click()
    wait_for_text(widget, 'Device: Headphones [-]')
    click()
    widget.scroll_down()
    click()
    wait_for_text(widget, 'Device: Earbuds [*]')
    click()
    wait_for_text(widget, 'Device: Earbuds [-]')
    widget.scroll_down()
    widget.scroll_down()
    assert text() == 'BT Headphones, Speaker'

def test_adapter_actions(bluetooth_manager):
    if False:
        return 10
    widget = bluetooth_manager.c.widget['bluetooth']

    def text():
        if False:
            print('Hello World!')
        return widget.info()['text']

    def click():
        if False:
            return 10
        bluetooth_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    wait_for_text(widget, 'BT Speaker')
    widget.scroll_up()
    assert text() == f'Adapter: {ADAPTER_NAME} [*]'
    click()
    wait_for_text(widget, 'Turn power off')
    click()
    wait_for_text(widget, 'Turn power on')
    widget.scroll_up()
    assert text() == 'Turn discovery on'
    click()
    wait_for_text(widget, 'Turn discovery off')
    click()
    wait_for_text(widget, 'Turn discovery on')
    widget.scroll_up()
    assert text() == 'Exit'
    click()
    wait_for_text(widget, f'Adapter: {ADAPTER_NAME} [-]')

@pytest.mark.parametrize('bluetooth_manager', [{'symbol_connected': 'C', 'symbol_paired': 'P', 'symbol_unknown': 'U', 'symbol_powered': ('ON', 'OFF')}], indirect=True)
def test_custom_symbols(bluetooth_manager):
    if False:
        i = 10
        return i + 15
    widget = bluetooth_manager.c.widget['bluetooth']

    def text():
        if False:
            while True:
                i = 10
        return widget.info()['text']

    def click():
        if False:
            i = 10
            return i + 15
        bluetooth_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, 1)
    wait_for_text(widget, 'BT Speaker')
    widget.scroll_up()
    assert text() == f'Adapter: {ADAPTER_NAME} [ON]'
    click()
    wait_for_text(widget, 'Turn power off')
    click()
    widget.scroll_up()
    widget.scroll_up()
    click()
    wait_for_text(widget, f'Adapter: {ADAPTER_NAME} [OFF]')
    widget.scroll_up()
    assert text() == 'Device: Earbuds [U]'
    widget.scroll_up()
    assert text() == 'Device: Headphones [P]'
    widget.scroll_up()
    assert text() == 'Device: Speaker (75.0%) [C]'

@pytest.mark.parametrize('bluetooth_manager', [{'default_show_battery': True}], indirect=True)
def test_default_show_battery(bluetooth_manager):
    if False:
        while True:
            i = 10
    widget = bluetooth_manager.c.widget['bluetooth']
    wait_for_text(widget, 'BT Speaker (75.0%)')

@pytest.mark.parametrize('bluetooth_manager', [{'adapter_paths': ['/org/bluez/hci1']}], indirect=True)
def test_missing_adapter(bluetooth_manager):
    if False:
        i = 10
        return i + 15
    widget = bluetooth_manager.c.widget['bluetooth']

    def text():
        if False:
            return 10
        return widget.info()['text']
    wait_for_text(widget, 'BT ')
    widget.scroll_up()
    assert text() == 'BT '

@pytest.mark.parametrize('bluetooth_manager', [{'default_text': 'BT {connected_devices} {num_connected_devices} {adapters} {num_adapters}'}], indirect=True)
def test_default_text(bluetooth_manager):
    if False:
        for i in range(10):
            print('nop')
    widget = bluetooth_manager.c.widget['bluetooth']
    wait_for_text(widget, 'BT Speaker 1 qtile_bluez 1')

@pytest.mark.parametrize('bluetooth_manager', [{'hci': '/dev_22_22_22_22_22_22_22_22'}, {'device': '/dev_22_22_22_22_22_22_22_22'}], indirect=True)
def test_default_device(bluetooth_manager):
    if False:
        return 10
    widget = bluetooth_manager.c.widget['bluetooth']
    wait_for_text(widget, 'Device: Headphones [-]')