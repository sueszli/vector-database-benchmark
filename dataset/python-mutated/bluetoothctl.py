"""jc - JSON Convert `bluetoothctl` command output parser

Supports the following `bluetoothctl` subcommands:
- `bluetoothctl list`
- `bluetoothctl show`
- `bluetoothctl show <ctrl>`
- `bluetoothctl devices`
- `bluetoothctl info <dev>`

Usage (cli):

    $ bluetoothctl info <dev> | jc --bluetoothctl
or

    $ jc bluetoothctl info <dev>

Usage (module):

    import jc
    result = jc.parse('bluetoothctl', bluetoothctl_command_output)

Schema:

Because bluetoothctl is handling two main entities, controllers and devices,
the schema is shared between them. Most of the fields are common between
a controller and a device but there might be fields corresponding to one entity.

    Controller:
    [
        {
            "name":                 string,
            "is_default":           boolean,
            "is_public":            boolean,
            "is_random":            boolean,
            "address":              string,
            "alias":                string,
            "class":                string,
            "powered":              string,
            "discoverable":         string,
            "discoverable_timeout": string,
            "pairable":             string,
            "modalias":             string,
            "discovering":          string,
            "uuids":                array
        }
    ]

    Device:
    [
        {
            "name":                 string,
            "is_public":            boolean,
            "is_random":            boolean,
            "address":              string,
            "alias":                string,
            "appearance":           string,
            "class":                string,
            "icon":                 string,
            "paired":               string,
            "bonded":               string,
            "trusted":              string,
            "blocked":              string,
            "connected":            string,
            "legacy_pairing":       string,
            "rssi":                 int,
            "txpower":              int,
            "uuids":                array,
            "modalias":             string
        }
    ]

Examples:

    $ bluetoothctl info EB:06:EF:62:B3:19 | jc --bluetoothctl -p
    [
        {
            "address": "22:06:33:62:B3:19",
            "is_public": true,
            "name": "TaoTronics TT-BH336",
            "alias": "TaoTronics TT-BH336",
            "class": "0x00240455",
            "icon": "audio-headset",
            "paired": "no",
            "bonded": "no",
            "trusted": "no",
            "blocked": "no",
            "connected": "no",
            "legacy_pairing": "no",
            "uuids": [
                "Advanced Audio Distribu.. (0000120d-0000-1000-8000-00805f9b34fb)",
                "Audio Sink                (0000130b-0000-1000-8000-00805f9b34fb)",
                "A/V Remote Control        (0000140e-0000-1000-8000-00805f9b34fb)",
                "A/V Remote Control Cont.. (0000150f-0000-1000-8000-00805f9b34fb)",
                "Handsfree                 (0000161e-0000-1000-8000-00805f9b34fb)",
                "Headset                   (00001708-0000-1000-8000-00805f9b34fb)",
                "Headset HS                (00001831-0000-1000-8000-00805f9b34fb)"
            ],
            "rssi": -52,
            "txpower": 4
        }
    ]
"""
import re
from typing import List, Dict, Optional, Any
from jc.jc_types import JSONDictType
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.1'
    description = '`bluetoothctl` command parser'
    author = 'Jake Ob'
    author_email = 'iakopap at gmail.com'
    compatible = ['linux']
    magic_commands = ['bluetoothctl']
    tags = ['command']
__version__ = info.version
try:
    from typing import TypedDict
    Controller = TypedDict('Controller', {'name': str, 'is_default': bool, 'is_public': bool, 'is_random': bool, 'address': str, 'alias': str, 'class': str, 'powered': str, 'discoverable': str, 'discoverable_timeout': str, 'pairable': str, 'modalias': str, 'discovering': str, 'uuids': List[str]})
    Device = TypedDict('Device', {'name': str, 'is_public': bool, 'is_random': bool, 'address': str, 'alias': str, 'appearance': str, 'class': str, 'icon': str, 'paired': str, 'bonded': str, 'trusted': str, 'blocked': str, 'connected': str, 'legacy_pairing': str, 'rssi': int, 'txpower': int, 'uuids': List[str], 'modalias': str})
except ImportError:
    Controller = Dict[str, Any]
    Device = Dict[str, Any]
_controller_head_pattern = 'Controller (?P<address>([0-9A-F]{2}:){5}[0-9A-F]{2}) (?P<name>.+)'
_controller_line_pattern = '(\\s*Name:\\s*(?P<name>.+)' + '|\\s*Alias:\\s*(?P<alias>.+)' + '|\\s*Class:\\s*(?P<class>.+)' + '|\\s*Powered:\\s*(?P<powered>.+)' + '|\\s*Discoverable:\\s*(?P<discoverable>.+)' + '|\\s*DiscoverableTimeout:\\s*(?P<discoverable_timeout>.+)' + '|\\s*Pairable:\\s*(?P<pairable>.+)' + '|\\s*Modalias:\\s*(?P<modalias>.+)' + '|\\s*Discovering:\\s*(?P<discovering>.+)' + '|\\s*UUID:\\s*(?P<uuid>.+))'

def _parse_controller(next_lines: List[str]) -> Optional[Controller]:
    if False:
        i = 10
        return i + 15
    next_line = next_lines.pop()
    result = re.match(_controller_head_pattern, next_line)
    if not result:
        next_lines.append(next_line)
        return None
    matches = result.groupdict()
    name = matches['name']
    if name.endswith('not available'):
        return None
    controller: Controller = {'name': '', 'is_default': False, 'is_public': False, 'is_random': False, 'address': matches['address'], 'alias': '', 'class': '', 'powered': '', 'discoverable': '', 'discoverable_timeout': '', 'pairable': '', 'modalias': '', 'discovering': '', 'uuids': []}
    if name.endswith('[default]'):
        controller['is_default'] = True
        name = name.replace('[default]', '')
    elif name.endswith('(public)'):
        controller['is_public'] = True
        name = name.replace('(public)', '')
    elif name.endswith('(random)'):
        controller['is_random'] = True
        name = name.replace('(random)', '')
    controller['name'] = name.strip()
    while next_lines:
        next_line = next_lines.pop()
        result = re.match(_controller_line_pattern, next_line)
        if not result:
            next_lines.append(next_line)
            return controller
        matches = result.groupdict()
        if matches['name']:
            controller['name'] = matches['name']
        elif matches['alias']:
            controller['alias'] = matches['alias']
        elif matches['class']:
            controller['class'] = matches['class']
        elif matches['powered']:
            controller['powered'] = matches['powered']
        elif matches['discoverable']:
            controller['discoverable'] = matches['discoverable']
        elif matches['discoverable_timeout']:
            controller['discoverable_timeout'] = matches['discoverable_timeout']
        elif matches['pairable']:
            controller['pairable'] = matches['pairable']
        elif matches['modalias']:
            controller['modalias'] = matches['modalias']
        elif matches['discovering']:
            controller['discovering'] = matches['discovering']
        elif matches['uuid']:
            if not 'uuids' in controller:
                controller['uuids'] = []
            controller['uuids'].append(matches['uuid'])
    return controller
_device_head_pattern = 'Device (?P<address>([0-9A-F]{2}:){5}[0-9A-F]{2}) (?P<name>.+)'
_device_line_pattern = '(\\s*Name:\\s*(?P<name>.+)' + '|\\s*Alias:\\s*(?P<alias>.+)' + '|\\s*Appearance:\\s*(?P<appearance>.+)' + '|\\s*Class:\\s*(?P<class>.+)' + '|\\s*Icon:\\s*(?P<icon>.+)' + '|\\s*Paired:\\s*(?P<paired>.+)' + '|\\s*Bonded:\\s*(?P<bonded>.+)' + '|\\s*Trusted:\\s*(?P<trusted>.+)' + '|\\s*Blocked:\\s*(?P<blocked>.+)' + '|\\s*Connected:\\s*(?P<connected>.+)' + '|\\s*LegacyPairing:\\s*(?P<legacy_pairing>.+)' + '|\\s*Modalias:\\s*(?P<modalias>.+)' + '|\\s*RSSI:\\s*(?P<rssi>.+)' + '|\\s*TxPower:\\s*(?P<txpower>.+)' + '|\\s*UUID:\\s*(?P<uuid>.+))'

def _parse_device(next_lines: List[str], quiet: bool) -> Optional[Device]:
    if False:
        for i in range(10):
            print('nop')
    next_line = next_lines.pop()
    result = re.match(_device_head_pattern, next_line)
    if not result:
        next_lines.append(next_line)
        return None
    matches = result.groupdict()
    name = matches['name']
    if name.endswith('not available'):
        return None
    device: Device = {'name': '', 'is_public': False, 'is_random': False, 'address': matches['address'], 'alias': '', 'appearance': '', 'class': '', 'icon': '', 'paired': '', 'bonded': '', 'trusted': '', 'blocked': '', 'connected': '', 'legacy_pairing': '', 'rssi': 0, 'txpower': 0, 'uuids': [], 'modalias': ''}
    if name.endswith('(public)'):
        device['is_public'] = True
        name = name.replace('(public)', '')
    elif name.endswith('(random)'):
        device['is_random'] = True
        name = name.replace('(random)', '')
    device['name'] = name.strip()
    while next_lines:
        next_line = next_lines.pop()
        result = re.match(_device_line_pattern, next_line)
        if not result:
            next_lines.append(next_line)
            return device
        matches = result.groupdict()
        if matches['name']:
            device['name'] = matches['name']
        elif matches['alias']:
            device['alias'] = matches['alias']
        elif matches['appearance']:
            device['appearance'] = matches['appearance']
        elif matches['class']:
            device['class'] = matches['class']
        elif matches['icon']:
            device['icon'] = matches['icon']
        elif matches['paired']:
            device['paired'] = matches['paired']
        elif matches['bonded']:
            device['bonded'] = matches['bonded']
        elif matches['trusted']:
            device['trusted'] = matches['trusted']
        elif matches['blocked']:
            device['blocked'] = matches['blocked']
        elif matches['connected']:
            device['connected'] = matches['connected']
        elif matches['legacy_pairing']:
            device['legacy_pairing'] = matches['legacy_pairing']
        elif matches['rssi']:
            rssi = matches['rssi']
            try:
                device['rssi'] = int(rssi)
            except ValueError:
                if not quiet:
                    jc.utils.warning_message([f'{next_line} : rssi - {rssi} is not int-able'])
        elif matches['txpower']:
            txpower = matches['txpower']
            try:
                device['txpower'] = int(txpower)
            except ValueError:
                if not quiet:
                    jc.utils.warning_message([f'{next_line} : txpower - {txpower} is not int-able'])
        elif matches['uuid']:
            if not 'uuids' in device:
                device['uuids'] = []
            device['uuids'].append(matches['uuid'])
        elif matches['modalias']:
            device['modalias'] = matches['modalias']
    return device

def parse(data: str, raw: bool=False, quiet: bool=False) -> List[JSONDictType]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    result: List = []
    if jc.utils.has_data(data):
        linedata = data.splitlines()
        linedata.reverse()
        while linedata:
            element = None
            if data.startswith('Controller'):
                element = _parse_controller(linedata)
            elif data.startswith('Device'):
                element = _parse_device(linedata, quiet)
            if element:
                result.append(element)
            else:
                break
    return result