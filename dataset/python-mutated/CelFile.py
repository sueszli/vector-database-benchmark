"""Reading information from Affymetrix CEL files version 3 and 4."""
import struct
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.Affy.CelFile') from None

class ParserError(ValueError):
    """Affymetrix parser error."""

    def __init__(self, *args):
        if False:
            print('Hello World!')
        'Initialise class.'
        super().__init__(*args)

class Record:
    """Stores the information in a cel file.

    Example usage:

    >>> from Bio.Affy import CelFile
    >>> with open("Affy/affy_v3_example.CEL") as handle:
    ...     c = CelFile.read(handle)
    ...
    >>> print(c.ncols, c.nrows)
    5 5
    >>> print(c.intensities)
    [[   234.    170.  22177.    164.  22104.]
     [   188.    188.  21871.    168.  21883.]
     [   188.    193.  21455.    198.  21300.]
     [   188.    182.  21438.    188.  20945.]
     [   193.  20370.    174.  20605.    168.]]
    >>> print(c.stdevs)
    [[   24.     34.5  2669.     19.7  3661.2]
     [   29.8    29.8  2795.9    67.9  2792.4]
     [   29.8    88.7  2976.5    62.   2914.5]
     [   29.8    76.2  2759.5    49.2  2762. ]
     [   38.8  2611.8    26.6  2810.7    24.1]]
    >>> print(c.npix)
    [[25 25 25 25 25]
     [25 25 25 25 25]
     [25 25 25 25 25]
     [25 25 25 25 25]
     [25 25 25 25 25]]

    """

    def __init__(self):
        if False:
            return 10
        'Initialize the class.'
        self.version = None
        self.GridCornerUL = None
        self.GridCornerUR = None
        self.GridCornerLR = None
        self.GridCornerLL = None
        self.DatHeader = None
        self.Algorithm = None
        self.AlgorithmParameters = None
        self.NumberCells = None
        self.intensities = None
        self.stdevs = None
        self.npix = None
        self.nrows = None
        self.ncols = None
        self.nmask = None
        self.mask = None
        self.noutliers = None
        self.outliers = None
        self.modified = None

def read(handle, version=None):
    if False:
        return 10
    'Read Affymetrix CEL file and return Record object.\n\n    CEL files format versions 3 and 4 are supported.\n    Please specify the CEL file format as 3 or 4 if known for the version\n    argument. If the version number is not specified, the parser will attempt\n    to detect the version from the file contents.\n\n    The Record object returned by this function stores the intensities from\n    the CEL file in record.intensities.\n    Currently, record.mask and record.outliers are not set in when parsing\n    version 4 CEL files.\n\n    Example Usage:\n\n    >>> from Bio.Affy import CelFile\n    >>> with open("Affy/affy_v3_example.CEL") as handle:\n    ...     record = CelFile.read(handle)\n    ...\n    >>> record.version == 3\n    True\n    >>> print("%i by %i array" % record.intensities.shape)\n    5 by 5 array\n\n    >>> with open("Affy/affy_v4_example.CEL", "rb") as handle:\n    ...     record = CelFile.read(handle, version=4)\n    ...\n    >>> record.version == 4\n    True\n    >>> print("%i by %i array" % record.intensities.shape)\n    5 by 5 array\n\n    '
    try:
        data = handle.read(0)
    except AttributeError:
        raise ValueError('handle should be a file handle') from None
    data = handle.read(4)
    if not data:
        raise ValueError('Empty file.')
    if data == b'[CEL':
        raise ValueError('CEL file in version 3 format should be opened in text mode')
    if data == '[CEL':
        data += next(handle)
        if data.strip() != '[CEL]':
            raise ValueError('Failed to parse Affy Version 3 CEL file.')
        line = next(handle)
        (keyword, value) = line.split('=', 1)
        if keyword != 'Version':
            raise ValueError('Failed to parse Affy Version 3 CEL file.')
        version = int(value)
        if version != 3:
            raise ValueError('Incorrect version number in Affy Version 3 CEL file.')
        return _read_v3(handle)
    try:
        magicNumber = struct.unpack('<i', data)
    except TypeError:
        raise ValueError('CEL file in version 4 format should be opened in binary mode') from None
    except struct.error:
        raise ValueError('Failed to read magic number from Affy Version 4 CEL file') from None
    if magicNumber != (64,):
        raise ValueError('Incorrect magic number in Affy Version 4 CEL file')
    return _read_v4(handle)

def _read_v4(f):
    if False:
        i = 10
        return i + 15
    record = Record()
    preHeaders = ['version', 'columns', 'rows', 'cellNo', 'headerLen']
    preHeadersMap = {}
    headersMap = {}
    preHeadersMap['magic'] = 64
    try:
        for name in preHeaders:
            preHeadersMap[name] = struct.unpack('<i', f.read(4))[0]
    except struct.error:
        raise ParserError('Failed to parse CEL version 4 file') from None
    char = f.read(preHeadersMap['headerLen'])
    header = char.decode('ascii', 'ignore')
    for line in header.split('\n'):
        if '=' in line:
            headline = line.split('=')
            headersMap[headline[0]] = '='.join(headline[1:])
    record.version = preHeadersMap['version']
    if record.version != 4:
        raise ParserError('Incorrect version number in CEL version 4 file')
    record.GridCornerUL = headersMap['GridCornerUL']
    record.GridCornerUR = headersMap['GridCornerUR']
    record.GridCornerLR = headersMap['GridCornerLR']
    record.GridCornerLL = headersMap['GridCornerLL']
    record.DatHeader = headersMap['DatHeader']
    record.Algorithm = headersMap['Algorithm']
    record.AlgorithmParameters = headersMap['AlgorithmParameters']
    record.NumberCells = preHeadersMap['cellNo']
    record.nrows = int(headersMap['Rows'])
    record.ncols = int(headersMap['Cols'])
    record.nmask = None
    record.mask = None
    record.noutliers = None
    record.outliers = None
    record.modified = None

    def raiseBadHeader(field, expected):
        if False:
            return 10
        actual = int(headersMap[field])
        message = f'The header {field} is expected to be 0, not {actual}'
        if actual != expected:
            raise ParserError(message)
    raiseBadHeader('Axis-invertX', 0)
    raiseBadHeader('AxisInvertY', 0)
    raiseBadHeader('OffsetX', 0)
    raiseBadHeader('OffsetY', 0)
    char = b'\x00'
    safetyValve = 10 ** 4
    for i in range(safetyValve):
        char = f.read(1)
        if char == b'\x04':
            break
        if i == safetyValve:
            raise ParserError('Parse Error. The parser expects a short, undocumented binary blob terminating with ASCII EOF, x04')
    padding = f.read(15)
    structa = struct.Struct('< f f h')
    structSize = 10
    record.intensities = np.empty(record.NumberCells, dtype=float)
    record.stdevs = np.empty(record.NumberCells, dtype=float)
    record.npix = np.empty(record.NumberCells, dtype=int)
    b = f.read(structSize * record.NumberCells)
    for i in range(record.NumberCells):
        binaryFragment = b[i * structSize:(i + 1) * structSize]
        (intensity, stdevs, npix) = structa.unpack(binaryFragment)
        record.intensities[i] = intensity
        record.stdevs[i] = stdevs
        record.npix[i] = npix

    def reshape(array):
        if False:
            for i in range(10):
                print('nop')
        view = array.view()
        view.shape = (record.nrows, record.ncols)
        return view
    record.intensities = reshape(record.intensities)
    record.stdevs = reshape(record.stdevs)
    record.npix = reshape(record.npix)
    return record

def _read_v3(handle):
    if False:
        i = 10
        return i + 15
    record = Record()
    record.version = 3
    section = ''
    for line in handle:
        line = line.rstrip('\r\n')
        if not line:
            continue
        if line.startswith('[HEADER]'):
            section = 'HEADER'
        elif line.startswith('[INTENSITY]'):
            section = 'INTENSITY'
            record.intensities = np.zeros((record.nrows, record.ncols))
            record.stdevs = np.zeros((record.nrows, record.ncols))
            record.npix = np.zeros((record.nrows, record.ncols), int)
        elif line.startswith('[MASKS]'):
            section = 'MASKS'
            record.mask = np.zeros((record.nrows, record.ncols), bool)
        elif line.startswith('[OUTLIERS]'):
            section = 'OUTLIERS'
            record.outliers = np.zeros((record.nrows, record.ncols), bool)
        elif line.startswith('[MODIFIED]'):
            section = 'MODIFIED'
            record.modified = np.zeros((record.nrows, record.ncols))
        elif line.startswith('['):
            raise ParserError('Unknown section found in version 3 CEL file')
        elif section == 'HEADER':
            (key, value) = line.split('=', 1)
            if key == 'Cols':
                record.ncols = int(value)
            elif key == 'Rows':
                record.nrows = int(value)
            elif key == 'GridCornerUL':
                (x, y) = value.split()
                record.GridCornerUL = (int(x), int(y))
            elif key == 'GridCornerUR':
                (x, y) = value.split()
                record.GridCornerUR = (int(x), int(y))
            elif key == 'GridCornerLR':
                (x, y) = value.split()
                record.GridCornerLR = (int(x), int(y))
            elif key == 'GridCornerLL':
                (x, y) = value.split()
                record.GridCornerLL = (int(x), int(y))
            elif key == 'DatHeader':
                record.DatHeader = {}
                i = value.find(':')
                if i >= 0:
                    (min_max_pixel_intensity, filename) = value[:i].split()
                    record.DatHeader['filename'] = filename
                    assert min_max_pixel_intensity[0] == '['
                    assert min_max_pixel_intensity[-1] == ']'
                    (min_pixel_intensity, max_pixel_intensity) = min_max_pixel_intensity[1:-1].split('..')
                    record.DatHeader['min-pixel_intensity'] = int(min_pixel_intensity)
                    record.DatHeader['max-pixel_intensity'] = int(max_pixel_intensity)
                    value = value[i + 1:]
                    index = 0
                    field = value[index:index + 9]
                    if field[:4] != 'CLS=' or field[8] != ' ':
                        raise ValueError("Field does not start with 'CLS=' or have a blank space at position 8")
                    record.DatHeader['CLS'] = int(field[4:8])
                    index += 9
                    field = value[index:index + 9]
                    if field[:4] != 'RWS=' or field[8] != ' ':
                        raise ValueError("Field does not start with 'RWS=' or have a blank space at position 8")
                    record.DatHeader['RWS'] = int(field[4:8])
                    index += 9
                    field = value[index:index + 7]
                    if field[:4] != 'XIN=' or field[6] != ' ':
                        raise ValueError("Field does not start with 'XIN=' or have a blank space at position 6")
                    record.DatHeader['XIN'] = int(field[4:6])
                    index += 7
                    field = value[index:index + 7]
                    if field[:4] != 'YIN=' or field[6] != ' ':
                        raise ValueError("Field does not start with 'YIN=' or have a blank space at poition 6")
                    record.DatHeader['YIN'] = int(field[4:6])
                    index += 7
                    field = value[index:index + 6]
                    if field[:3] != 'VE=' or field[5] != ' ':
                        raise ValueError("Field does not start with 'VE=' or have a blank space at position 5")
                    record.DatHeader['VE'] = int(field[3:5])
                    index += 6
                    field = value[index:index + 7]
                    if field[6] != ' ':
                        raise ValueError("Field value for position 6 isn't a blank space")
                    temperature = field[:6].strip()
                    if temperature:
                        record.DatHeader['temperature'] = int(temperature)
                    else:
                        record.DatHeader['temperature'] = None
                    index += 7
                    field = value[index:index + 4]
                    if not field.endswith(' '):
                        raise ValueError("Field doesn't end with a blank space")
                    record.DatHeader['laser-power'] = float(field)
                    index += 4
                    field = value[index:index + 18]
                    if field[8] != ' ':
                        raise ValueError("Field value for position 8 isn't a blank space")
                    record.DatHeader['scan-date'] = field[:8]
                    if field[17] != ' ':
                        raise ValueError("Field value for position 17 isn't a blank space")
                    record.DatHeader['scan-date'] = field[:8]
                    record.DatHeader['scan-time'] = field[9:17]
                    index += 18
                    value = value[index:]
                subfields = value.split('\x14')
                if len(subfields) != 12:
                    ValueError("Subfields length isn't 12")
                subfield = subfields[0]
                try:
                    (scanner_id, scanner_type) = subfield.split()
                except ValueError:
                    scanner_id = subfield.strip()
                else:
                    record.DatHeader['scanner-type'] = scanner_type
                record.DatHeader['scanner-id'] = scanner_id
                record.DatHeader['array-type'] = subfields[2].strip()
                field = subfields[7].strip()
                if field:
                    record.DatHeader['filter-wavelength'] = int(field)
                field = subfields[8].strip()
                if field:
                    record.DatHeader['arc-radius'] = float(field)
                field = subfields[9].strip()
                if field:
                    record.DatHeader['laser-spotsize'] = float(field)
                field = subfields[10].strip()
                if field:
                    record.DatHeader['pixel-size'] = float(field)
                field = subfields[11].strip()
                if field:
                    record.DatHeader['image-orientation'] = int(field)
            elif key == 'Algorithm':
                record.Algorithm = value
            elif key == 'AlgorithmParameters':
                parameters = value.split(';')
                values = {}
                for parameter in parameters:
                    (key, value) = parameter.split(':', 1)
                    if key in ('Percentile', 'CellMargin', 'FullFeatureWidth', 'FullFeatureHeight', 'PoolWidthExtenstion', 'PoolHeightExtension', 'NumPixelsToUse', 'ExtendPoolWidth', 'ExtendPoolHeight', 'OutlierRatioLowPercentile', 'OutlierRatioHighPercentile', 'HalfCellRowsDivisor', 'HalfCellRowsRemainder', 'HighCutoff', 'LowCutoff', 'featureRows', 'featureColumns'):
                        values[key] = int(value)
                    elif key in ('OutlierHigh', 'OutlierLow', 'StdMult', 'PercentileSpread', 'PairCutoff', 'featureWidth', 'featureHeight'):
                        values[key] = float(value)
                    elif key in ('FixedCellSize', 'IgnoreOutliersInShiftRows', 'FeatureExtraction', 'UseSubgrids', 'RandomizePixels', 'ImageCalibration', 'IgnoreShiftRowOutliers'):
                        if value == 'TRUE':
                            value = True
                        elif value == 'FALSE':
                            value = False
                        else:
                            raise ValueError('Unexpected boolean value')
                        values[key] = value
                    elif key in ('AlgVersion', 'ErrorBasis', 'CellIntensityCalculationType'):
                        values[key] = value
                    else:
                        raise ValueError('Unexpected tag in AlgorithmParameters')
                record.AlgorithmParameters = values
        elif section == 'INTENSITY':
            if line.startswith('NumberCells='):
                (key, value) = line.split('=', 1)
                record.NumberCells = int(value)
            elif line.startswith('CellHeader='):
                (key, value) = line.split('=', 1)
                if value.split() != ['X', 'Y', 'MEAN', 'STDV', 'NPIXELS']:
                    raise ParserError('Unexpected CellHeader in INTENSITY section CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.intensities[x, y] = float(words[2])
                record.stdevs[x, y] = float(words[3])
                record.npix[x, y] = int(words[4])
        elif section == 'MASKS':
            if line.startswith('NumberCells='):
                (key, value) = line.split('=', 1)
                record.nmask = int(value)
            elif line.startswith('CellHeader='):
                (key, value) = line.split('=', 1)
                if value.split() != ['X', 'Y']:
                    raise ParserError('Unexpected CellHeader in MASKS section in CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.mask[x, y] = True
        elif section == 'OUTLIERS':
            if line.startswith('NumberCells='):
                (key, value) = line.split('=', 1)
                record.noutliers = int(value)
            elif line.startswith('CellHeader='):
                (key, value) = line.split('=', 1)
                if value.split() != ['X', 'Y']:
                    raise ParserError('Unexpected CellHeader in OUTLIERS section in CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.outliers[x, y] = True
        elif section == 'MODIFIED':
            if line.startswith('NumberCells='):
                (key, value) = line.split('=', 1)
                record.nmodified = int(value)
            elif line.startswith('CellHeader='):
                (key, value) = line.split('=', 1)
                if value.split() != ['X', 'Y', 'ORIGMEAN']:
                    raise ParserError('Unexpected CellHeader in MODIFIED section in CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.modified[x, y] = float(words[2])
    return record
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()