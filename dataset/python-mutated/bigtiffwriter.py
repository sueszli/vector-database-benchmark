import os
import numpy as np
from .npimg import NpImage
from ..checkdeps import HAS_GDAL, HAS_PIL, HAS_IMGIO
if HAS_GDAL:
    from osgeo import gdal

class BigTiffWriter:
    """
	This class is designed to write a bigtif with jpeg compression
	writing a large tiff file without trigger a memory overflow is possible with the help of GDAL library
	jpeg compression allows to maintain a reasonable file size
	transparency or nodata are stored in an internal tiff mask because it's not possible to have an alpha channel when using jpg compression
	"""

    def __del__(self):
        if False:
            print('Hello World!')
        self.ds = None

    def __init__(self, path, w, h, georef, geoTiffOptions={'TFW': 'YES', 'TILED': 'YES', 'BIGTIFF': 'YES', 'COMPRESS': 'JPEG', 'JPEG_QUALITY': 80, 'PHOTOMETRIC': 'YCBCR'}):
        if False:
            return 10
        '\n\t\tpath = fule system path for the ouput tiff\n\t\tw, h = width and height in pixels\n\t\tgeoref : a Georef object used to set georeferencing informations, optional\n\t\tgeoTiffOptions : GDAL create option for tiff format\n\t\t'
        if not HAS_GDAL:
            raise ImportError('GDAL interface unavailable')
        self.w = w
        self.h = h
        self.size = (w, h)
        self.path = path
        self.georef = georef
        if geoTiffOptions.get('COMPRESS', None) == 'JPEG':
            self.useMask = True
            gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
            n = 3
        else:
            self.useMask = False
            n = 4
        self.nbBands = n
        options = [str(k) + '=' + str(v) for (k, v) in geoTiffOptions.items()]
        driver = gdal.GetDriverByName('GTiff')
        gdtype = gdal.GDT_Byte
        self.dtype = 'uint8'
        self.ds = driver.Create(path, w, h, n, gdtype, options)
        if self.useMask:
            self.ds.CreateMaskBand(gdal.GMF_PER_DATASET)
            self.mask = self.ds.GetRasterBand(1).GetMaskBand()
            self.mask.Fill(255)
        elif n == 4:
            self.ds.GetRasterBand(4).Fill(255)
        self.ds.SetGeoTransform(self.georef.toGDAL())
        if self.georef.crs is not None:
            self.ds.SetProjection(self.georef.crs.getOgrSpatialRef().ExportToWkt())

    def paste(self, data, x, y):
        if False:
            return 10
        'data = numpy array or NpImg'
        img = NpImage(data)
        data = img.data
        for bandIdx in range(3):
            bandArray = data[:, :, bandIdx]
            self.ds.GetRasterBand(bandIdx + 1).WriteArray(bandArray, x, y)
        hasAlpha = data.shape[2] == 4
        if hasAlpha:
            alpha = data[:, :, 3]
            if self.useMask:
                self.mask.WriteArray(alpha, x, y)
            else:
                self.ds.GetRasterBand(4).WriteArray(alpha, x, y)
        else:
            pass
            '\n\t\t\t#make alpha band or internal mask fully opaque\n\t\t\th, w = data.shape[0], data.shape[1]\n\t\t\talpha = np.full((h, w), 255, np.uint8)\n\t\t\tif self.useMask:\n\t\t\t\tself.mask.WriteArray(alpha, x, y)\n\t\t\telse:\n\t\t\t\tself.ds.GetRasterBand(4).WriteArray(alpha, x, y)\n\t\t\t'

    def __repr__(self):
        if False:
            return 10
        return '\n'.join(['* Data infos :', ' size {}'.format(self.size), ' type {}'.format(self.dtype), ' number of bands {}'.format(self.nbBands), '* Georef & Geometry : \n{}'.format(self.georef)])