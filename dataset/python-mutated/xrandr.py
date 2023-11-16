"""jc - JSON Convert `xrandr` command output parser

Usage (cli):

    $ xrandr | jc --xrandr
    $ xrandr --properties | jc --xrandr

or

    $ jc xrandr

Usage (module):

    import jc
    result = jc.parse('xrandr', xrandr_command_output)

Schema:

    {
      "screens": [
        {
          "screen_number":                     integer,
          "minimum_width":                     integer,
          "minimum_height":                    integer,
          "current_width":                     integer,
          "current_height":                    integer,
          "maximum_width":                     integer,
          "maximum_height":                    integer,
          "devices": {
            "modes": [
              {
                "resolution_width":            integer,
                "resolution_height":           integer,
                "is_high_resolution":          boolean,
                "frequencies": [
                  {
                    "frequency":               float,
                    "is_current":              boolean,
                    "is_preferred":            boolean
                  }
                ]
              }
            ]
          },
          "is_connected":                      boolean,
          "is_primary":                        boolean,
          "device_name":                       string,
          "model_name":                        string,
          "product_id"                         string,
          "serial_number":                     string,
          "resolution_width":                  integer,
          "resolution_height":                 integer,
          "offset_width":                      integer,
          "offset_height":                     integer,
          "dimension_width":                   integer,
          "dimension_height":                  integer,
          "rotation":                          string,
          "reflection":                        string
        }
      ],
    }

Examples:

    $ xrandr | jc --xrandr -p
    {
      "screens": [
        {
          "screen_number": 0,
          "minimum_width": 8,
          "minimum_height": 8,
          "current_width": 1920,
          "current_height": 1080,
          "maximum_width": 32767,
          "maximum_height": 32767,
          "devices": {
            "modes": [
              {
                "resolution_width": 1920,
                "resolution_height": 1080,
                "is_high_resolution": false,
                "frequencies": [
                  {
                    "frequency": 60.03,
                    "is_current": true,
                    "is_preferred": true
                  },
                  {
                    "frequency": 59.93,
                    "is_current": false,
                    "is_preferred": false
                  }
                ]
              },
              {
                "resolution_width": 1680,
                "resolution_height": 1050,
                "is_high_resolution": false,
                "frequencies": [
                  {
                    "frequency": 59.88,
                    "is_current": false,
                    "is_preferred": false
                  }
                ]
              }
            ],
            "is_connected": true,
            "is_primary": true,
            "device_name": "eDP1",
            "resolution_width": 1920,
            "resolution_height": 1080,
            "offset_width": 0,
            "offset_height": 0,
            "dimension_width": 310,
            "dimension_height": 170,
            "rotation": "normal",
            "reflection": "normal"
          }
        }
      ]
    }

    $ xrandr --properties | jc --xrandr -p
    {
      "screens": [
        {
          "screen_number": 0,
          "minimum_width": 8,
          "minimum_height": 8,
          "current_width": 1920,
          "current_height": 1080,
          "maximum_width": 32767,
          "maximum_height": 32767,
          "devices": {
            "modes": [
              {
                "resolution_width": 1920,
                "resolution_height": 1080,
                "is_high_resolution": false,
                "frequencies": [
                  {
                    "frequency": 60.03,
                    "is_current": true,
                    "is_preferred": true
                  },
                  {
                    "frequency": 59.93,
                    "is_current": false,
                    "is_preferred": false
                  }
                ]
              },
              {
                "resolution_width": 1680,
                "resolution_height": 1050,
                "is_high_resolution": false,
                "frequencies": [
                  {
                    "frequency": 59.88,
                    "is_current": false,
                    "is_preferred": false
                  }
                ]
              }
            ],
            "is_connected": true,
            "is_primary": true,
            "device_name": "eDP1",
            "model_name": "ASUS VW193S",
            "product_id": "54297",
            "serial_number": "78L8021107",
            "resolution_width": 1920,
            "resolution_height": 1080,
            "offset_width": 0,
            "offset_height": 0,
            "dimension_width": 310,
            "dimension_height": 170,
            "rotation": "normal",
            "reflection": "normal"
          }
        }
      ]
    }
"""
import re
from typing import Dict, List, Optional, Union
import jc.utils
from jc.parsers.pyedid.edid import Edid
from jc.parsers.pyedid.helpers.edid_helper import EdidHelper

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.3'
    description = '`xrandr` command parser'
    author = 'Kevin Lyter'
    author_email = 'code (at) lyterk.com'
    details = 'Using parts of the pyedid library at https://github.com/jojonas/pyedid.'
    compatible = ['linux', 'darwin', 'cygwin', 'aix', 'freebsd']
    magic_commands = ['xrandr']
    tags = ['command']
__version__ = info.version
try:
    from typing import TypedDict
    Frequency = TypedDict('Frequency', {'frequency': float, 'is_current': bool, 'is_preferred': bool})
    Mode = TypedDict('Mode', {'resolution_width': int, 'resolution_height': int, 'is_high_resolution': bool, 'frequencies': List[Frequency]})
    Model = TypedDict('Model', {'name': str, 'product_id': str, 'serial_number': str})
    Device = TypedDict('Device', {'device_name': str, 'model_name': str, 'product_id': str, 'serial_number': str, 'is_connected': bool, 'is_primary': bool, 'resolution_width': int, 'resolution_height': int, 'offset_width': int, 'offset_height': int, 'dimension_width': int, 'dimension_height': int, 'modes': List[Mode], 'rotation': str, 'reflection': str})
    Screen = TypedDict('Screen', {'screen_number': int, 'minimum_width': int, 'minimum_height': int, 'current_width': int, 'current_height': int, 'maximum_width': int, 'maximum_height': int, 'devices': List[Device]})
    Response = TypedDict('Response', {'screens': List[Screen]})
except ImportError:
    Screen = Dict[str, Union[int, str]]
    Device = Dict[str, Union[str, int, bool]]
    Frequency = Dict[str, Union[float, bool]]
    Mode = Dict[str, Union[int, bool, List[Frequency]]]
    Model = Dict[str, str]
    Response = Dict[str, Union[Device, Mode, Screen]]
_screen_pattern = 'Screen (?P<screen_number>\\d+): ' + 'minimum (?P<minimum_width>\\d+) x (?P<minimum_height>\\d+), ' + 'current (?P<current_width>\\d+) x (?P<current_height>\\d+), ' + 'maximum (?P<maximum_width>\\d+) x (?P<maximum_height>\\d+)'

def _parse_screen(next_lines: List[str]) -> Optional[Screen]:
    if False:
        while True:
            i = 10
    next_line = next_lines.pop()
    result = re.match(_screen_pattern, next_line)
    if not result:
        next_lines.append(next_line)
        return None
    raw_matches = result.groupdict()
    screen: Screen = {'devices': []}
    for (k, v) in raw_matches.items():
        screen[k] = int(v)
    while next_lines:
        device: Optional[Device] = _parse_device(next_lines)
        if not device:
            break
        else:
            screen['devices'].append(device)
    return screen
_device_pattern = '(?P<device_name>.+) ' + '(?P<is_connected>(connected|disconnected)) ?' + '(?P<is_primary> primary)? ?' + '((?P<resolution_width>\\d+)x(?P<resolution_height>\\d+)' + '\\+(?P<offset_width>\\d+)\\+(?P<offset_height>\\d+))? ' + '(?P<rotation>(normal|right|left|inverted)?) ?' + '(?P<reflection>(X axis|Y axis|X and Y axis)?) ?' + '\\(normal left inverted right x axis y axis\\)' + '( ((?P<dimension_width>\\d+)mm x (?P<dimension_height>\\d+)mm)?)?'

def _parse_device(next_lines: List[str], quiet: bool=False) -> Optional[Device]:
    if False:
        i = 10
        return i + 15
    if not next_lines:
        return None
    next_line = next_lines.pop()
    result = re.match(_device_pattern, next_line)
    if not result:
        next_lines.append(next_line)
        return None
    matches = result.groupdict()
    device: Device = {'modes': [], 'is_connected': matches['is_connected'] == 'connected', 'is_primary': matches['is_primary'] is not None and len(matches['is_primary']) > 0, 'device_name': matches['device_name'], 'rotation': matches['rotation'] or 'normal', 'reflection': matches['reflection'] or 'normal'}
    for (k, v) in matches.items():
        if k not in {'is_connected', 'is_primary', 'device_name', 'rotation', 'reflection'}:
            try:
                if v:
                    device[k] = int(v)
            except ValueError:
                if not quiet:
                    jc.utils.warning_message([f'{next_line} : {k} - {v} is not int-able'])
    model: Optional[Model] = _parse_model(next_lines, quiet)
    if model:
        device['model_name'] = model['name']
        device['product_id'] = model['product_id']
        device['serial_number'] = model['serial_number']
    while next_lines:
        next_line = next_lines.pop()
        next_mode: Optional[Mode] = _parse_mode(next_line)
        if next_mode:
            device['modes'].append(next_mode)
        elif re.match(_device_pattern, next_line):
            next_lines.append(next_line)
            break
    return device
_edid_head_pattern = '\\s*EDID:\\s*'
_edid_line_pattern = '\\s*(?P<edid_line>[0-9a-fA-F]{32})\\s*'

def _parse_model(next_lines: List[str], quiet: bool=False) -> Optional[Model]:
    if False:
        for i in range(10):
            print('nop')
    if not next_lines:
        return None
    next_line = next_lines.pop()
    if not re.match(_edid_head_pattern, next_line):
        next_lines.append(next_line)
        return None
    edid_hex_value = ''
    while next_lines:
        next_line = next_lines.pop()
        result = re.match(_edid_line_pattern, next_line)
        if not result:
            next_lines.append(next_line)
            break
        matches = result.groupdict()
        edid_hex_value += matches['edid_line']
    edid = Edid(EdidHelper.hex2bytes(edid_hex_value))
    model: Model = {'name': edid.name or 'Generic', 'product_id': str(edid.product), 'serial_number': str(edid.serial)}
    return model
_mode_pattern = '\\s*(?P<resolution_width>\\d+)x(?P<resolution_height>\\d+)(?P<is_high_resolution>i)?\\s+(?P<rest>.*)'
_frequencies_pattern = '(((?P<frequency>\\d+\\.\\d+)(?P<star>\\*| |)(?P<plus>\\+?)?)+)'

def _parse_mode(line: str) -> Optional[Mode]:
    if False:
        print('Hello World!')
    result = re.match(_mode_pattern, line)
    frequencies: List[Frequency] = []
    if not result:
        return None
    d = result.groupdict()
    resolution_width = int(d['resolution_width'])
    resolution_height = int(d['resolution_height'])
    is_high_resolution = d['is_high_resolution'] is not None
    mode: Mode = {'resolution_width': resolution_width, 'resolution_height': resolution_height, 'is_high_resolution': is_high_resolution, 'frequencies': frequencies}
    result = re.finditer(_frequencies_pattern, d['rest'])
    if not result:
        return mode
    for match in result:
        d = match.groupdict()
        frequency = float(d['frequency'])
        is_current = len(d['star'].strip()) > 0
        is_preferred = len(d['plus'].strip()) > 0
        f: Frequency = {'frequency': frequency, 'is_current': is_current, 'is_preferred': is_preferred}
        mode['frequencies'].append(f)
    return mode

def parse(data: str, raw: bool=False, quiet: bool=False) -> Dict:
    if False:
        return 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    linedata = data.splitlines()
    linedata.reverse()
    result: Response = {'screens': []}
    if jc.utils.has_data(data):
        while linedata:
            screen = _parse_screen(linedata)
            if screen:
                result['screens'].append(screen)
    return result