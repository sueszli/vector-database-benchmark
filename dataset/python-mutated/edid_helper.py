"""
EDID helper
"""
from subprocess import CalledProcessError, check_output
from typing import ByteString, List
__all__ = ['EdidHelper']

class EdidHelper:
    """Class for working with EDID data"""

    @staticmethod
    def hex2bytes(hex_data: str) -> ByteString:
        if False:
            i = 10
            return i + 15
        'Convert hex EDID string to bytes\n\n        Args:\n            hex_data (str): hex edid string\n\n        Returns:\n            ByteString: edid byte string\n        '
        if len(hex_data) > 256:
            hex_data = hex_data[:256]
        numbers = []
        for i in range(0, len(hex_data), 2):
            pair = hex_data[i:i + 2]
            numbers.append(int(pair, 16))
        return bytes(numbers)

    @classmethod
    def get_edids(cls) -> List[ByteString]:
        if False:
            print('Hello World!')
        'Get edids from xrandr\n\n        Raises:\n            `RuntimeError`: if error with retrieving xrandr util data\n\n        Returns:\n            List[ByteString]: list with edids\n        '
        try:
            output = check_output(['xrandr', '--verbose'])
        except (CalledProcessError, FileNotFoundError) as err:
            raise RuntimeError('Error retrieving xrandr util data: {}'.format(err)) from None
        edids = []
        lines = output.splitlines()
        for (i, line) in enumerate(lines):
            line = line.decode().strip()
            if line.startswith('EDID:'):
                selection = lines[i + 1:i + 9]
                selection = list((s.decode().strip() for s in selection))
                selection = ''.join(selection)
                bytes_section = cls.hex2bytes(selection)
                edids.append(bytes_section)
        return edids