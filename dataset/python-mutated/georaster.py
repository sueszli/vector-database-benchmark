import os
import logging
log = logging.getLogger(__name__)
from ..lib import Tyf
from .georef import GeoRef
from .npimg import NpImage
from .img_utils import getImgFormat, getImgDim
from ..utils import XY as xy
from ..errors import OverlapError
from ..checkdeps import HAS_GDAL
if HAS_GDAL:
    from osgeo import gdal

class GeoRaster:
    """A class to represent a georaster file"""

    def __init__(self, path, subBoxGeo=None, useGDAL=False):
        if False:
            print('Hello World!')
        '\n\t\tsubBoxGeo : a BBOX object in CRS coordinate space\n\t\tuseGDAL : use GDAL (if available) for extract raster informations\n\t\t'
        self.path = path
        self.wfPath = self._getWfPath()
        self.format = None
        self.size = None
        self.depth = None
        self.dtype = None
        self.nbBands = None
        self.noData = None
        self.georef = None
        if not useGDAL or not HAS_GDAL:
            self.format = getImgFormat(path)
            if self.format not in ['TIFF', 'BMP', 'PNG', 'JPEG', 'JPEG2000']:
                raise IOError('Unsupported format {}'.format(self.format))
            if self.isTiff:
                self._fromTIFF()
                if not self.isGeoref and self.hasWorldFile:
                    self.georef = GeoRef.fromWorldFile(self.wfPath, self.size)
                else:
                    pass
            else:
                (w, h) = getImgDim(self.path)
                if w is None or h is None:
                    raise IOError('Unable to read raster size')
                else:
                    self.size = xy(w, h)
                if self.hasWorldFile:
                    self.georef = GeoRef.fromWorldFile(self.wfPath, self.size)
        else:
            self._fromGDAL()
        if not self.isGeoref:
            raise IOError('Unable to read georef infos from worldfile or geotiff tags')
        if subBoxGeo is not None:
            self.georef.setSubBoxGeo(subBoxGeo)

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        return getattr(self.georef, attr)

    def _getWfPath(self):
        if False:
            return 10
        'Try to find a worlfile path for this raster'
        ext = self.path[-3:].lower()
        extTest = []
        extTest.append(ext[0] + ext[2] + 'w')
        extTest.append(extTest[0] + 'x')
        extTest.append(ext + 'w')
        extTest.append('wld')
        extTest.extend([ext.upper() for ext in extTest])
        for wfExt in extTest:
            pathTest = self.path[0:len(self.path) - 3] + wfExt
            if os.path.isfile(pathTest):
                return pathTest
        return None

    def _fromTIFF(self):
        if False:
            print('Hello World!')
        'Use Tyf to extract raster infos from geotiff tags'
        if not self.isTiff or not self.fileExists:
            return
        tif = Tyf.open(self.path)[0]
        self.size = xy(tif['ImageWidth'], tif['ImageLength'])
        self.nbBands = tif['SamplesPerPixel']
        self.depth = tif['BitsPerSample']
        if self.nbBands > 1:
            self.depth = self.depth[0]
        sampleFormatMap = {1: 'uint', 2: 'int', 3: 'float', None: 'uint', 6: 'complex'}
        try:
            self.dtype = sampleFormatMap[tif['SampleFormat']]
        except KeyError:
            self.dtype = 'uint'
        try:
            self.noData = float(tif['GDAL_NODATA'])
        except KeyError:
            self.noData = None
        try:
            self.georef = GeoRef.fromTyf(tif)
        except Exception as e:
            log.warning('Cannot extract georefencing informations from tif tags')
            pass

    def _fromGDAL(self):
        if False:
            return 10
        'Use GDAL to extract raster infos and init'
        if self.path is None or not self.fileExists:
            raise IOError('Cannot find file on disk')
        ds = gdal.Open(self.path, gdal.GA_ReadOnly)
        self.size = xy(ds.RasterXSize, ds.RasterYSize)
        self.format = ds.GetDriver().ShortName
        if self.format in ['JP2OpenJPEG', 'JP2ECW', 'JP2KAK', 'JP2MrSID']:
            self.format = 'JPEG2000'
        self.nbBands = ds.RasterCount
        b1 = ds.GetRasterBand(1)
        self.noData = b1.GetNoDataValue()
        ddtype = gdal.GetDataTypeName(b1.DataType)
        if ddtype == 'Byte':
            self.dtype = 'uint'
            self.depth = 8
        else:
            self.dtype = ddtype[0:len(ddtype) - 2].lower()
            self.depth = int(ddtype[-2:])
        self.georef = GeoRef.fromGDAL(ds)
        (ds, b1) = (None, None)

    @property
    def fileExists(self):
        if False:
            i = 10
            return i + 15
        'Test if the file exists on disk'
        return os.path.isfile(self.path)

    @property
    def baseName(self):
        if False:
            return 10
        if self.path is not None:
            (folder, fileName) = os.path.split(self.path)
            (baseName, ext) = os.path.splitext(fileName)
            return baseName

    @property
    def isTiff(self):
        if False:
            i = 10
            return i + 15
        'Flag if the image format is TIFF'
        if self.format in ['TIFF', 'GTiff']:
            return True
        else:
            return False

    @property
    def hasWorldFile(self):
        if False:
            i = 10
            return i + 15
        return self.wfPath is not None

    @property
    def isGeoref(self):
        if False:
            while True:
                i = 10
        'Flag if georef parameters have been extracted'
        if self.georef is not None:
            if self.origin is not None and self.pxSize is not None and (self.rotation is not None):
                return True
            else:
                return False
        else:
            return False

    @property
    def isOneBand(self):
        if False:
            return 10
        return self.nbBands == 1

    @property
    def isFloat(self):
        if False:
            return 10
        return self.dtype in ['Float', 'float']

    @property
    def ddtype(self):
        if False:
            return 10
        "\n\t\tGet data type and depth in a concatenate string like\n\t\t'int8', 'int16', 'uint16', 'int32', 'uint32', 'float32' ...\n\t\tCan be used to define numpy or gdal data type\n\t\t"
        if self.dtype is None or self.depth is None:
            return None
        else:
            return self.dtype + str(self.depth)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '\n'.join(['* Paths infos :', ' path {}'.format(self.path), ' worldfile {}'.format(self.wfPath), ' format {}'.format(self.format), '* Data infos :', ' size {}'.format(self.size), ' bit depth {}'.format(self.depth), ' data type {}'.format(self.dtype), ' number of bands {}'.format(self.nbBands), ' nodata value {}'.format(self.noData), '* Georef & Geometry : \n{}'.format(self.georef)])

    def toGDAL(self):
        if False:
            return 10
        'Get GDAL dataset'
        return gdal.Open(self.path, gdal.GA_ReadOnly)

    def readAsNpArray(self, subset=True):
        if False:
            print('Hello World!')
        'Read raster pixels values as Numpy Array'
        if subset and self.subBoxGeo is not None:
            img = NpImage(self.path, subBoxPx=self.subBoxPx, noData=self.noData, georef=self.georef, adjustGeoref=True)
        else:
            img = NpImage(self.path, noData=self.noData, georef=self.georef)
        return img