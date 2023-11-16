"""
Edid module
"""
import struct
from collections import namedtuple
from typing import ByteString
__all__ = ['Edid']

class Edid:
    """Edid class

    Raises:
        `ValueError`: if invalid edid data
    """
    _STRUCT_FORMAT = '<8sHHIBBBBBBBBB10sHB16s18s18s18s18sBB'
    _TIMINGS = {0: (1280, 1024, 75.0), 1: (1024, 768, 75.0), 2: (1024, 768, 70.0), 3: (1024, 768, 60.0), 4: (1024, 768, 87.0), 5: (832, 624, 75.0), 6: (800, 600, 75.0), 7: (800, 600, 72.0), 8: (800, 600, 60.0), 9: (800, 600, 56.0), 10: (640, 480, 75.0), 11: (640, 480, 72.0), 12: (640, 480, 67.0), 13: (640, 480, 60.0), 14: (720, 400, 88.0), 15: (720, 400, 70.0)}
    _ASPECT_RATIOS = {0: (16, 10), 1: (4, 3), 2: (5, 4), 3: (16, 9)}
    _RawEdid = namedtuple('RawEdid', ('header', 'manu_id', 'prod_id', 'serial_no', 'manu_week', 'manu_year', 'edid_version', 'edid_revision', 'input_type', 'width', 'height', 'gamma', 'features', 'color', 'timings_supported', 'timings_reserved', 'timings_edid', 'timing_1', 'timing_2', 'timing_3', 'timing_4', 'extension', 'checksum'))

    def __init__(self, edid: ByteString):
        if False:
            print('Hello World!')
        self._parse_edid(edid)

    def _parse_edid(self, edid: ByteString):
        if False:
            while True:
                i = 10
        'Convert edid byte string to edid object'
        if struct.calcsize(self._STRUCT_FORMAT) != 128:
            raise ValueError('Wrong edid size.')
        if sum(map(int, edid)) % 256 != 0:
            raise ValueError('Checksum mismatch.')
        unpacked = struct.unpack(self._STRUCT_FORMAT, edid)
        raw_edid = self._RawEdid(*unpacked)
        if raw_edid.header != b'\x00\xff\xff\xff\xff\xff\xff\x00':
            raise ValueError('Invalid header.')
        self.raw = edid
        self.manufacturer_id = raw_edid.manu_id
        self.product = raw_edid.prod_id
        self.year = raw_edid.manu_year + 1990
        self.edid_version = '{:d}.{:d}'.format(raw_edid.edid_version, raw_edid.edid_revision)
        self.type = 'digital' if raw_edid.input_type & 255 else 'analog'
        self.width = float(raw_edid.width)
        self.height = float(raw_edid.height)
        self.gamma = (raw_edid.gamma + 100) / 100
        self.dpms_standby = bool(raw_edid.features & 255)
        self.dpms_suspend = bool(raw_edid.features & 127)
        self.dpms_activeoff = bool(raw_edid.features & 63)
        self.resolutions = []
        for i in range(16):
            bit = raw_edid.timings_supported & 1 << i
            if bit:
                self.resolutions.append(self._TIMINGS[i])
        for i in range(8):
            bytes_data = raw_edid.timings_edid[2 * i:2 * i + 2]
            if bytes_data == b'\x01\x01':
                continue
            (byte1, byte2) = bytes_data
            x_res = 8 * (int(byte1) + 31)
            aspect_ratio = self._ASPECT_RATIOS[byte2 >> 6 & 3]
            y_res = int(x_res * aspect_ratio[1] / aspect_ratio[0])
            rate = (int(byte2) & 63) + 60.0
            self.resolutions.append((x_res, y_res, rate))
        self.name = None
        self.serial = None
        for timing_bytes in (raw_edid.timing_1, raw_edid.timing_2, raw_edid.timing_3, raw_edid.timing_4):
            if timing_bytes[0:2] == b'\x00\x00':
                timing_type = timing_bytes[3]
                if timing_type in (255, 254, 252):
                    buffer = timing_bytes[5:]
                    buffer = buffer.partition(b'\n')[0]
                    text = buffer.decode('cp437')
                    if timing_type == 255:
                        self.serial = text
                    elif timing_type == 252:
                        self.name = text
        if not self.serial:
            self.serial = raw_edid.serial_no

    def __repr__(self):
        if False:
            return 10
        clsname = self.__class__.__name__
        attributes = []
        for name in dir(self):
            if not name.startswith('_'):
                value = getattr(self, name)
                attributes.append('\t{}={}'.format(name, value))
        return '{}(\n{}\n)'.format(clsname, ', \n'.join(attributes))