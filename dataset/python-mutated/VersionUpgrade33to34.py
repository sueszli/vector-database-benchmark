import configparser
import io
from typing import Dict, List, Tuple
from UM.VersionUpgrade import VersionUpgrade
_renamed_settings = {'infill_hollow': 'infill_support_enabled'}

class VersionUpgrade33to34(VersionUpgrade):

    def upgradeInstanceContainer(self, serialized: str, filename: str) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(serialized)
        parser['general']['version'] = '4'
        if 'values' in parser:
            if 'infill_hollow' in parser['values'] and parser['values']['infill_hollow'] and ('support_angle' in parser['values']):
                parser['values']['infill_support_angle'] = parser['values']['support_angle']
            for (original, replacement) in _renamed_settings.items():
                if original in parser['values']:
                    parser['values'][replacement] = parser['values'][original]
                    del parser['values'][original]
        result = io.StringIO()
        parser.write(result)
        return ([filename], [result.getvalue()])