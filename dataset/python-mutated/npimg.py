import os
import io
import random
import numpy as np
from .georef import GeoRef
from ..proj.reproj import reprojImg
from ..maths.fillnodata import replace_nans
from ..utils import XY as xy
from ..checkdeps import HAS_GDAL, HAS_PIL, HAS_IMGIO
from .. import settings
if HAS_PIL:
    from PIL import Image
if HAS_GDAL:
    from osgeo import gdal
if HAS_IMGIO:
    from ..lib import imageio

class NpImage:
    """Represent an image as Numpy array"""

    def _getIFACE(self):
        if False:
            while True:
                i = 10
        engine = settings.img_engine
        if engine == 'AUTO':
            if HAS_GDAL:
                return 'GDAL'
            elif HAS_IMGIO:
                return 'IMGIO'
            elif HAS_PIL:
                return 'PIL'
            else:
                raise ImportError('No image engine available')
        elif engine == 'GDAL' and HAS_GDAL:
            return 'GDAL'
        elif engine == 'IMGIO' and HAS_IMGIO:
            return 'IMGIO'
        elif engine == 'PIL' and HAS_PIL:
            return 'PIL'
        else:
            raise ImportError(str(engine) + ' interface unavailable')

    def __getattr__(self, attr):
        if False:
            return 10
        if self.isGeoref:
            return getattr(self.georef, attr)
        else:
            raise AttributeError(str(type(self)) + 'object has no attribute' + str(attr))

    def __init__(self, data, subBoxPx=None, noData=None, georef=None, adjustGeoref=False):
        if False:
            print('Hello World!')
        '\n\t\tinit from file path, bytes data, Numpy array, NpImage, PIL Image or GDAL dataset\n\t\tsubBoxPx : a BBOX object in pixel coordinates space used as data filter (will by applyed) (y counting from top)\n\t\tnoData : the value used to represent nodata, will be used to define a numpy mask\n\t\tgeoref : a Georef object used to set georeferencing informations, optional\n\t\tadjustGeoref: determine if the submited georef must be adjusted against the subbox or if its already correct\n\n\t\tNotes :\n\t\t* With GDAL the subbox filter can be applyed at reading level whereas with others imaging\n\t\tlibrary, all the data must be extracted before we can extract the subset (using numpy slice).\n\t\tIn this case, the dataset must fit entirely in memory otherwise it will raise an overflow error\n\t\t* If no georef was submited and when the class is init using gdal support or from another npImage instance,\n\t\texisting georef of input data will be automatically extracted and adjusted against the subbox\n\t\t'
        self.IFACE = self._getIFACE()
        self.data = None
        self.subBoxPx = subBoxPx
        self.noData = noData
        self.georef = georef
        if self.subBoxPx is not None and self.georef is not None:
            if adjustGeoref:
                self.georef.setSubBoxPx(subBoxPx)
                self.georef.applySubBox()
        if isinstance(data, NpImage):
            self.data = self._applySubBox(data.data)
            if data.isGeoref and (not self.isGeoref):
                self.georef = data.georef
                if self.subBoxPx is not None:
                    self.georef.setSubBoxPx(subBoxPx)
                    self.georef.applySubBox()
        if isinstance(data, np.ndarray):
            self.data = self._applySubBox(data)
        if isinstance(data, bytes):
            self.data = self._npFromBLOB(data)
        if isinstance(data, str):
            if os.path.exists(data):
                self.data = self._npFromPath(data)
            else:
                raise ValueError('Unable to load image data')
        if HAS_GDAL:
            if isinstance(data, gdal.Dataset):
                self.data = self._npFromGDAL(data)
        if HAS_PIL:
            if Image.isImageType(data):
                self.data = self._npFromPIL(data)
        if self.data is None:
            raise ValueError('Unable to load image data')
        if self.noData is not None:
            self.data = np.ma.masked_array(self.data, self.data == self.noData)

    @property
    def size(self):
        if False:
            print('Hello World!')
        return xy(self.data.shape[1], self.data.shape[0])

    @property
    def isGeoref(self):
        if False:
            i = 10
            return i + 15
        'Flag if georef parameters have been extracted'
        if self.georef is not None:
            return True
        else:
            return False

    @property
    def nbBands(self):
        if False:
            i = 10
            return i + 15
        if len(self.data.shape) == 2:
            return 1
        elif len(self.data.shape) == 3:
            return self.data.shape[2]

    @property
    def hasAlpha(self):
        if False:
            i = 10
            return i + 15
        return self.nbBands == 4

    @property
    def isOneBand(self):
        if False:
            i = 10
            return i + 15
        return self.nbBands == 1

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        "return string ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64']"
        return self.data.dtype

    @property
    def isFloat(self):
        if False:
            while True:
                i = 10
        if self.dtype in ['float16', 'float32', 'float64']:
            return True
        else:
            return False

    def getMin(self, bandIdx=0):
        if False:
            i = 10
            return i + 15
        if self.nbBands == 1:
            return self.data.min()
        else:
            return self.data[:, :, bandIdx].min()

    def getMax(self, bandIdx=0):
        if False:
            return 10
        if self.nbBands == 1:
            return self.data.max()
        else:
            return self.data[:, :, bandIdx].max()

    @classmethod
    def new(cls, w, h, bkgColor=(255, 255, 255, 255), noData=None, georef=None):
        if False:
            print('Hello World!')
        (r, g, b, a) = bkgColor
        data = np.empty((h, w, 4), np.uint8)
        data[:, :, 0] = r
        data[:, :, 1] = g
        data[:, :, 2] = b
        data[:, :, 3] = a
        return cls(data, noData=noData, georef=georef)

    def _applySubBox(self, data):
        if False:
            while True:
                i = 10
        'Use numpy slice to extract subset of data'
        if self.subBoxPx is not None:
            (x1, x2) = (self.subBoxPx.xmin, self.subBoxPx.xmax + 1)
            (y1, y2) = (self.subBoxPx.ymin, self.subBoxPx.ymax + 1)
            if len(data.shape) == 2:
                data = data[y1:y2, x1:x2]
            else:
                data = data[y1:y2, x1:x2, :]
            self.subBoxPx = None
        return data

    def _npFromPath(self, path):
        if False:
            return 10
        'Get Numpy array from a file path'
        if self.IFACE == 'PIL':
            img = Image.open(path)
            return self._npFromPIL(img)
        elif self.IFACE == 'IMGIO':
            return self._npFromImgIO(path)
        elif self.IFACE == 'GDAL':
            ds = gdal.Open(path)
            return self._npFromGDAL(ds)

    def _npFromBLOB(self, data):
        if False:
            i = 10
            return i + 15
        'Get Numpy array from Bytes data'
        if self.IFACE == 'PIL':
            img = Image.open(io.BytesIO(data))
            data = self._npFromPIL(img)
        elif self.IFACE == 'IMGIO':
            img = io.BytesIO(data)
            data = self._npFromImgIO(img)
        elif self.IFACE == 'GDAL':
            vsipath = '/vsimem/' + ''.join((random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(5)))
            gdal.FileFromMemBuffer(vsipath, data)
            ds = gdal.Open(vsipath)
            data = self._npFromGDAL(ds)
            ds = None
            gdal.Unlink(vsipath)
        return data

    def _npFromImgIO(self, img):
        if False:
            return 10
        'Use ImageIO to extract numpy array from image path or bytesIO'
        data = imageio.imread(img)
        return self._applySubBox(data)

    def _npFromPIL(self, img):
        if False:
            for i in range(10):
                print('nop')
        'Get Numpy array from PIL Image instance'
        if img.mode == 'P':
            img = img.convert('RGBA')
        data = np.asarray(img)
        data.setflags(write=True)
        return self._applySubBox(data)

    def _npFromGDAL(self, ds):
        if False:
            print('Hello World!')
        'Get Numpy array from GDAL dataset instance'
        if self.subBoxPx is not None:
            (startx, starty) = (self.subBoxPx.xmin, self.subBoxPx.ymin)
            width = self.subBoxPx.xmax - self.subBoxPx.xmin + 1
            height = self.subBoxPx.ymax - self.subBoxPx.ymin + 1
            data = ds.ReadAsArray(startx, starty, width, height)
        else:
            data = ds.ReadAsArray()
        if len(data.shape) == 3:
            data = np.rollaxis(data, 0, 3)
        else:
            ctable = ds.GetRasterBand(1).GetColorTable()
            if ctable is not None:
                nbColors = ctable.GetCount()
                keys = np.array([i for i in range(nbColors)])
                values = np.array([ctable.GetColorEntry(i) for i in range(nbColors)])
                sortIdx = np.argsort(keys)
                idx = np.searchsorted(keys, data, sorter=sortIdx)
                data = values[sortIdx][idx]
        if not self.isGeoref:
            self.georef = GeoRef.fromGDAL(ds)
            if self.subBoxPx is not None and self.georef is not None:
                self.georef.applySubBox()
        return data

    def toBLOB(self, ext='PNG'):
        if False:
            while True:
                i = 10
        'Get bytes raw data'
        if ext == 'JPG':
            ext = 'JPEG'
        if self.IFACE == 'PIL':
            b = io.BytesIO()
            img = Image.fromarray(self.data)
            img.save(b, format=ext)
            data = b.getvalue()
        elif self.IFACE == 'IMGIO':
            if ext == 'JPEG' and self.hasAlpha:
                self.removeAlpha()
            data = imageio.imwrite(imageio.RETURN_BYTES, self.data, format=ext)
        elif self.IFACE == 'GDAL':
            mem = self.toGDAL()
            name = ''.join((random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(5)))
            vsiname = '/vsimem/' + name + '.png'
            out = gdal.GetDriverByName(ext).CreateCopy(vsiname, mem)
            f = gdal.VSIFOpenL(vsiname, 'rb')
            gdal.VSIFSeekL(f, 0, 2)
            size = gdal.VSIFTellL(f)
            gdal.VSIFSeekL(f, 0, 0)
            data = gdal.VSIFReadL(1, size, f)
            gdal.VSIFCloseL(f)
            gdal.Unlink(vsiname)
            mem = None
        return data

    def toPIL(self):
        if False:
            for i in range(10):
                print('nop')
        'Get PIL Image instance'
        return Image.fromarray(self.data)

    def toGDAL(self):
        if False:
            while True:
                i = 10
        'Get GDAL memory driver dataset'
        (w, h) = self.size
        n = self.nbBands
        dtype = str(self.dtype)
        if dtype == 'uint8':
            dtype = 'byte'
        dtype = gdal.GetDataTypeByName(dtype)
        mem = gdal.GetDriverByName('MEM').Create('', w, h, n, dtype)
        if self.isOneBand:
            mem.GetRasterBand(1).WriteArray(self.data)
        else:
            for bandIdx in range(n):
                bandArray = self.data[:, :, bandIdx]
                mem.GetRasterBand(bandIdx + 1).WriteArray(bandArray)
        if self.isGeoref:
            mem.SetGeoTransform(self.georef.toGDAL())
            if self.georef.crs is not None:
                mem.SetProjection(self.georef.crs.getOgrSpatialRef().ExportToWkt())
        return mem

    def removeAlpha(self):
        if False:
            while True:
                i = 10
        if self.hasAlpha:
            self.data = self.data[:, :, 0:3]

    def addAlpha(self, opacity=255):
        if False:
            for i in range(10):
                print('nop')
        if self.nbBands == 3:
            (w, h) = self.size
            alpha = np.empty((h, w), dtype=self.dtype)
            alpha.fill(opacity)
            alpha = np.expand_dims(alpha, axis=2)
            self.data = np.append(self.data, alpha, axis=2)

    def save(self, path):
        if False:
            print('Hello World!')
        '\n\t\tsave the numpy array to a new image file\n\t\toutput format is defined by path extension\n\t\t'
        imgFormat = path[-3:]
        if self.IFACE == 'PIL':
            self.toPIL().save(path)
        elif self.IFACE == 'IMGIO':
            if imgFormat == 'jpg' and self.hasAlpha:
                self.removeAlpha()
            imageio.imwrite(path, self.data)
        elif self.IFACE == 'GDAL':
            if imgFormat == 'png':
                driver = 'PNG'
            elif imgFormat == 'jpg':
                driver = 'JPEG'
            elif imgFormat == 'tif':
                driver = 'Gtiff'
            else:
                raise ValueError('Cannot write to ' + imgFormat + ' image format')
            mem = self.toGDAL()
            out = gdal.GetDriverByName(driver).CreateCopy(path, mem)
            mem = out = None
        if self.isGeoref:
            self.georef.toWorldFile(os.path.splitext(path)[0] + '.wld')

    def paste(self, data, x, y):
        if False:
            while True:
                i = 10
        img = NpImage(data)
        data = img.data
        (w, h) = img.size
        if img.isOneBand and self.isOneBand:
            self.data[y:y + h, x:x + w] = data
        elif not img.isOneBand and self.isOneBand or (img.isOneBand and (not self.isOneBand)):
            raise ValueError('Paste error, cannot mix one band with multiband')
        if self.hasAlpha:
            n = img.nbBands
            self.data[y:y + h, x:x + w, 0:n] = data
        else:
            n = self.nbBands
            self.data[y:y + h, x:x + w, :] = data[:, :, 0:n]

    def cast2float(self):
        if False:
            i = 10
            return i + 15
        if not self.isFloat:
            self.data = self.data.astype('float32')

    def fillNodata(self):
        if False:
            while True:
                i = 10
        if not np.ma.is_masked(self.data):
            return
        if self.IFACE == 'GDAL':
            (height, width) = self.data.shape
            ds = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GetDataTypeByName('float32'))
            b = ds.GetRasterBand(1)
            b.SetNoDataValue(self.noData)
            self.data = np.ma.filled(self.data, self.noData)
            b.WriteArray(self.data)
            gdal.FillNodata(targetBand=b, maskBand=None, maxSearchDist=max(self.size.xy), smoothingIterations=0)
            self.data = b.ReadAsArray()
            (ds, b) = (None, None)
        else:
            self.cast2float()
            self.data = np.ma.filled(self.data, np.NaN)
            self.data = replace_nans(self.data, max_iter=5, tolerance=0.5, kernel_size=2, method='localmean')

    def reproj(self, crs1, crs2, out_ul=None, out_size=None, out_res=None, sqPx=False, resamplAlg='BL'):
        if False:
            return 10
        ds1 = self.toGDAL()
        if not self.isGeoref:
            raise IOError('Unable to reproject non georeferenced image')
        ds2 = reprojImg(crs1, crs2, ds1, out_ul=out_ul, out_size=out_size, out_res=out_res, sqPx=sqPx, resamplAlg=resamplAlg)
        return NpImage(ds2)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '\n'.join(['* Data infos :', ' size {}'.format(self.size), ' type {}'.format(self.dtype), ' number of bands {}'.format(self.nbBands), ' nodata value {}'.format(self.noData), '* Statistics : min {} max {}'.format(self.getMin(), self.getMax()), '* Georef & Geometry : \n{}'.format(self.georef)])