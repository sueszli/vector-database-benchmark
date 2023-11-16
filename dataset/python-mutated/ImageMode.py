import sys
_modes = None

class ModeDescriptor:
    """Wrapper for mode strings."""

    def __init__(self, mode, bands, basemode, basetype, typestr):
        if False:
            for i in range(10):
                print('nop')
        self.mode = mode
        self.bands = bands
        self.basemode = basemode
        self.basetype = basetype
        self.typestr = typestr

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.mode

def getmode(mode):
    if False:
        print('Hello World!')
    'Gets a mode descriptor for the given mode.'
    global _modes
    if not _modes:
        modes = {}
        endian = '<' if sys.byteorder == 'little' else '>'
        for (m, (basemode, basetype, bands, typestr)) in {'1': ('L', 'L', ('1',), '|b1'), 'L': ('L', 'L', ('L',), '|u1'), 'I': ('L', 'I', ('I',), endian + 'i4'), 'F': ('L', 'F', ('F',), endian + 'f4'), 'P': ('P', 'L', ('P',), '|u1'), 'RGB': ('RGB', 'L', ('R', 'G', 'B'), '|u1'), 'RGBX': ('RGB', 'L', ('R', 'G', 'B', 'X'), '|u1'), 'RGBA': ('RGB', 'L', ('R', 'G', 'B', 'A'), '|u1'), 'CMYK': ('RGB', 'L', ('C', 'M', 'Y', 'K'), '|u1'), 'YCbCr': ('RGB', 'L', ('Y', 'Cb', 'Cr'), '|u1'), 'LAB': ('RGB', 'L', ('L', 'A', 'B'), '|u1'), 'HSV': ('RGB', 'L', ('H', 'S', 'V'), '|u1'), 'RGBa': ('RGB', 'L', ('R', 'G', 'B', 'a'), '|u1'), 'BGR;15': ('RGB', 'L', ('B', 'G', 'R'), '|u1'), 'BGR;16': ('RGB', 'L', ('B', 'G', 'R'), '|u1'), 'BGR;24': ('RGB', 'L', ('B', 'G', 'R'), '|u1'), 'LA': ('L', 'L', ('L', 'A'), '|u1'), 'La': ('L', 'L', ('L', 'a'), '|u1'), 'PA': ('RGB', 'L', ('P', 'A'), '|u1')}.items():
            modes[m] = ModeDescriptor(m, bands, basemode, basetype, typestr)
        for (i16mode, typestr) in {'I;16': '<u2', 'I;16S': '<i2', 'I;16L': '<u2', 'I;16LS': '<i2', 'I;16B': '>u2', 'I;16BS': '>i2', 'I;16N': endian + 'u2', 'I;16NS': endian + 'i2', 'I;32': '<u4', 'I;32B': '>u4', 'I;32L': '<u4', 'I;32S': '<i4', 'I;32BS': '>i4', 'I;32LS': '<i4'}.items():
            modes[i16mode] = ModeDescriptor(i16mode, ('I',), 'L', 'L', typestr)
        _modes = modes
    return _modes[mode]