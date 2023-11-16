"""
Support for Bluetooth (using BlueZ in Linux).

The following packages are required packages for this module:

    bluez >= 5.7
    bluez-libs >= 5.7
    bluez-utils >= 5.7
    pybluez >= 0.18
"""
import shlex
import salt.utils.validate.net
from salt.exceptions import CommandExecutionError
HAS_PYBLUEZ = False
try:
    import bluetooth
    HAS_PYBLUEZ = True
except ImportError:
    pass
__func_alias__ = {'address_': 'address'}
__virtualname__ = 'bluetooth'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load the module if bluetooth is installed\n    '
    if HAS_PYBLUEZ:
        return __virtualname__
    return (False, 'The bluetooth execution module cannot be loaded: bluetooth not installed.')

def version():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return Bluez version from bluetoothd -v\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetoothd.version\n    "
    cmd = 'bluetoothctl -v'
    out = __salt__['cmd.run'](cmd).splitlines()
    bluez_version = out[0]
    pybluez_version = '<= 0.18 (Unknown, but installed)'
    try:
        pybluez_version = bluetooth.__version__
    except Exception as exc:
        pass
    return {'Bluez': bluez_version, 'PyBluez': pybluez_version}

def address_():
    if False:
        i = 10
        return i + 15
    "\n    Get the many addresses of the Bluetooth adapter\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.address\n    "
    ret = {}
    cmd = 'hciconfig'
    out = __salt__['cmd.run'](cmd).splitlines()
    dev = ''
    for line in out:
        if line.startswith('hci'):
            comps = line.split(':')
            dev = comps[0]
            ret[dev] = {'device': dev, 'path': '/sys/class/bluetooth/{}'.format(dev)}
        if 'BD Address' in line:
            comps = line.split()
            ret[dev]['address'] = comps[2]
        if 'DOWN' in line:
            ret[dev]['power'] = 'off'
        if 'UP RUNNING' in line:
            ret[dev]['power'] = 'on'
    return ret

def power(dev, mode):
    if False:
        while True:
            i = 10
    "\n    Power a bluetooth device on or off\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.power hci0 on\n        salt '*' bluetooth.power hci0 off\n    "
    if dev not in address_():
        raise CommandExecutionError('Invalid dev passed to bluetooth.power')
    if mode == 'on' or mode is True:
        state = 'up'
        mode = 'on'
    else:
        state = 'down'
        mode = 'off'
    cmd = 'hciconfig {} {}'.format(dev, state)
    __salt__['cmd.run'](cmd).splitlines()
    info = address_()
    if info[dev]['power'] == mode:
        return True
    return False

def discoverable(dev):
    if False:
        return 10
    "\n    Enable this bluetooth device to be discoverable.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.discoverable hci0\n    "
    if dev not in address_():
        raise CommandExecutionError('Invalid dev passed to bluetooth.discoverable')
    cmd = 'hciconfig {} iscan'.format(dev)
    __salt__['cmd.run'](cmd).splitlines()
    cmd = 'hciconfig {}'.format(dev)
    out = __salt__['cmd.run'](cmd)
    if 'UP RUNNING ISCAN' in out:
        return True
    return False

def noscan(dev):
    if False:
        print('Hello World!')
    "\n    Turn off scanning modes on this device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.noscan hci0\n    "
    if dev not in address_():
        raise CommandExecutionError('Invalid dev passed to bluetooth.noscan')
    cmd = 'hciconfig {} noscan'.format(dev)
    __salt__['cmd.run'](cmd).splitlines()
    cmd = 'hciconfig {}'.format(dev)
    out = __salt__['cmd.run'](cmd)
    if 'SCAN' in out:
        return False
    return True

def scan():
    if False:
        print('Hello World!')
    "\n    Scan for bluetooth devices in the area\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.scan\n    "
    ret = []
    devices = bluetooth.discover_devices(lookup_names=True)
    for device in devices:
        ret.append({device[0]: device[1]})
    return ret

def block(bdaddr):
    if False:
        print('Hello World!')
    "\n    Block a specific bluetooth device by BD Address\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.block DE:AD:BE:EF:CA:FE\n    "
    if not salt.utils.validate.net.mac(bdaddr):
        raise CommandExecutionError('Invalid BD address passed to bluetooth.block')
    cmd = 'hciconfig {} block'.format(bdaddr)
    __salt__['cmd.run'](cmd).splitlines()

def unblock(bdaddr):
    if False:
        for i in range(10):
            print('nop')
    "\n    Unblock a specific bluetooth device by BD Address\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.unblock DE:AD:BE:EF:CA:FE\n    "
    if not salt.utils.validate.net.mac(bdaddr):
        raise CommandExecutionError('Invalid BD address passed to bluetooth.unblock')
    cmd = 'hciconfig {} unblock'.format(bdaddr)
    __salt__['cmd.run'](cmd).splitlines()

def pair(address, key):
    if False:
        print('Hello World!')
    "\n    Pair the bluetooth adapter with a device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.pair DE:AD:BE:EF:CA:FE 1234\n\n    Where DE:AD:BE:EF:CA:FE is the address of the device to pair with, and 1234\n    is the passphrase.\n\n    TODO: This function is currently broken, as the bluez-simple-agent program\n    no longer ships with BlueZ >= 5.0. It needs to be refactored.\n    "
    if not salt.utils.validate.net.mac(address):
        raise CommandExecutionError('Invalid BD address passed to bluetooth.pair')
    try:
        int(key)
    except Exception:
        raise CommandExecutionError('bluetooth.pair requires a numerical key to be used')
    addy = address_()
    cmd = 'echo {} | bluez-simple-agent {} {}'.format(shlex.quote(addy['device']), shlex.quote(address), shlex.quote(key))
    out = __salt__['cmd.run'](cmd, python_shell=True).splitlines()
    return out

def unpair(address):
    if False:
        return 10
    "\n    Unpair the bluetooth adapter from a device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.unpair DE:AD:BE:EF:CA:FE\n\n    Where DE:AD:BE:EF:CA:FE is the address of the device to unpair.\n\n    TODO: This function is currently broken, as the bluez-simple-agent program\n    no longer ships with BlueZ >= 5.0. It needs to be refactored.\n    "
    if not salt.utils.validate.net.mac(address):
        raise CommandExecutionError('Invalid BD address passed to bluetooth.unpair')
    cmd = 'bluez-test-device remove {}'.format(address)
    out = __salt__['cmd.run'](cmd).splitlines()
    return out

def start():
    if False:
        while True:
            i = 10
    "\n    Start the bluetooth service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.start\n    "
    out = __salt__['service.start']('bluetooth')
    return out

def stop():
    if False:
        print('Hello World!')
    "\n    Stop the bluetooth service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bluetooth.stop\n    "
    out = __salt__['service.stop']('bluetooth')
    return out