import struct
from django.core.exceptions import ValidationError
from .const import BANDTYPE_FLAG_HASNODATA, BANDTYPE_PIXTYPE_MASK, GDAL_TO_POSTGIS, GDAL_TO_STRUCT, POSTGIS_HEADER_STRUCTURE, POSTGIS_TO_GDAL, STRUCT_SIZE

def pack(structure, data):
    if False:
        while True:
            i = 10
    '\n    Pack data into hex string with little endian format.\n    '
    return struct.pack('<' + structure, *data)

def unpack(structure, data):
    if False:
        i = 10
        return i + 15
    '\n    Unpack little endian hexlified binary string into a list.\n    '
    return struct.unpack('<' + structure, bytes.fromhex(data))

def chunk(data, index):
    if False:
        print('Hello World!')
    '\n    Split a string into two parts at the input index.\n    '
    return (data[:index], data[index:])

def from_pgraster(data):
    if False:
        while True:
            i = 10
    '\n    Convert a PostGIS HEX String into a dictionary.\n    '
    if data is None:
        return
    (header, data) = chunk(data, 122)
    header = unpack(POSTGIS_HEADER_STRUCTURE, header)
    bands = []
    pixeltypes = []
    while data:
        (pixeltype_with_flags, data) = chunk(data, 2)
        pixeltype_with_flags = unpack('B', pixeltype_with_flags)[0]
        pixeltype = pixeltype_with_flags & BANDTYPE_PIXTYPE_MASK
        pixeltype = POSTGIS_TO_GDAL[pixeltype]
        pack_type = GDAL_TO_STRUCT[pixeltype]
        pack_size = 2 * STRUCT_SIZE[pack_type]
        (nodata, data) = chunk(data, pack_size)
        nodata = unpack(pack_type, nodata)[0]
        (band, data) = chunk(data, pack_size * header[10] * header[11])
        band_result = {'data': bytes.fromhex(band)}
        if pixeltype_with_flags & BANDTYPE_FLAG_HASNODATA:
            band_result['nodata_value'] = nodata
        bands.append(band_result)
        pixeltypes.append(pixeltype)
    if len(set(pixeltypes)) != 1:
        raise ValidationError('Band pixeltypes are not all equal.')
    return {'srid': int(header[9]), 'width': header[10], 'height': header[11], 'datatype': pixeltypes[0], 'origin': (header[5], header[6]), 'scale': (header[3], header[4]), 'skew': (header[7], header[8]), 'bands': bands}

def to_pgraster(rast):
    if False:
        print('Hello World!')
    '\n    Convert a GDALRaster into PostGIS Raster format.\n    '
    rasterheader = (1, 0, len(rast.bands), rast.scale.x, rast.scale.y, rast.origin.x, rast.origin.y, rast.skew.x, rast.skew.y, rast.srs.srid, rast.width, rast.height)
    result = pack(POSTGIS_HEADER_STRUCTURE, rasterheader)
    for band in rast.bands:
        structure = 'B' + GDAL_TO_STRUCT[band.datatype()]
        pixeltype = GDAL_TO_POSTGIS[band.datatype()]
        if band.nodata_value is not None:
            pixeltype |= BANDTYPE_FLAG_HASNODATA
        bandheader = pack(structure, (pixeltype, band.nodata_value or 0))
        result += bandheader + band.data(as_memoryview=True)
    return result