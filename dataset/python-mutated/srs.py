import logging
log = logging.getLogger(__name__)
from .utm import UTM, UTM_EPSG_CODES
from .srv import EPSGIO
from ..checkdeps import HAS_GDAL, HAS_PYPROJ
if HAS_GDAL:
    from osgeo import osr, gdal
if HAS_PYPROJ:
    import pyproj

class SRS:
    """
	A simple class to handle Spatial Ref System inputs
	"""

    @classmethod
    def validate(cls, crs):
        if False:
            print('Hello World!')
        try:
            cls(crs)
            return True
        except Exception as e:
            log.error('Cannot initialize crs', exc_info=True)
            return False

    def __init__(self, crs):
        if False:
            i = 10
            return i + 15
        '\n\t\tValid crs input can be :\n\t\t> an epsg code (integer or string)\n\t\t> a SRID string (AUTH:CODE)\n\t\t> a proj4 string\n\t\t'
        crs = str(crs)
        if crs.isdigit():
            self.auth = 'EPSG'
            self.code = int(crs)
            self.proj4 = '+init=epsg:' + str(self.code)
        elif ':' in crs:
            (self.auth, self.code) = crs.split(':')
            if self.code.isdigit():
                self.code = int(self.code)
                if self.auth.startswith('+init='):
                    (_, self.auth) = self.auth.split('=')
                self.auth = self.auth.upper()
                self.proj4 = '+init=' + self.auth.lower() + ':' + str(self.code)
            else:
                raise ValueError('Invalid CRS : ' + crs)
        elif all([param.startswith('+') for param in crs.split(' ') if param]):
            self.auth = None
            self.code = None
            self.proj4 = crs
        else:
            raise ValueError('Invalid CRS : ' + crs)

    @classmethod
    def fromGDAL(cls, ds):
        if False:
            return 10
        if not HAS_GDAL:
            raise ImportError('GDAL not available')
        wkt = ds.GetProjection()
        if not wkt:
            raise ImportError('This raster has no projection')
        crs = osr.SpatialReference()
        crs.ImportFromWkt(wkt)
        return cls(crs.ExportToProj4())

    @property
    def SRID(self):
        if False:
            print('Hello World!')
        if self.isSRID:
            return self.auth + ':' + str(self.code)
        else:
            return None

    @property
    def hasCode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.code is not None

    @property
    def hasAuth(self):
        if False:
            i = 10
            return i + 15
        return self.auth is not None

    @property
    def isSRID(self):
        if False:
            for i in range(10):
                print('nop')
        return self.hasAuth and self.hasCode

    @property
    def isEPSG(self):
        if False:
            print('Hello World!')
        return self.auth == 'EPSG' and self.code is not None

    @property
    def isWM(self):
        if False:
            for i in range(10):
                print('nop')
        return self.auth == 'EPSG' and self.code == 3857

    @property
    def isWGS84(self):
        if False:
            while True:
                i = 10
        return self.auth == 'EPSG' and self.code == 4326

    @property
    def isUTM(self):
        if False:
            while True:
                i = 10
        return self.auth == 'EPSG' and self.code in UTM_EPSG_CODES

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return the best string representation for this crs'
        if self.isSRID:
            return self.SRID
        else:
            return self.proj4

    def __eq__(self, srs2):
        if False:
            while True:
                i = 10
        return self.__str__() == srs2.__str__()

    def getOgrSpatialRef(self):
        if False:
            print('Hello World!')
        'Build gdal osr spatial ref object'
        if not HAS_GDAL:
            raise ImportError('GDAL not available')
        prj = osr.SpatialReference()
        if self.isEPSG:
            r = prj.ImportFromEPSG(self.code)
        else:
            r = prj.ImportFromProj4(self.proj4)
        if r > 0:
            raise ValueError('Cannot initialize osr : ' + self.proj4)
        return prj

    def getPyProj(self):
        if False:
            while True:
                i = 10
        'Build pyproj object'
        if not HAS_PYPROJ:
            raise ImportError('PYPROJ not available')
        if self.isSRID:
            return pyproj.Proj(self.SRID)
        else:
            try:
                return pyproj.Proj(self.proj4)
            except Exception as e:
                raise ValueError('Cannot initialize pyproj object for projection {}. Error : {}'.format(self.proj4, e))

    def loadProj4(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a Python dict of proj4 parameters'
        dc = {}
        if self.proj4 is None:
            return dc
        for param in self.proj4.split(' '):
            if param.count('=') == 1:
                (k, v) = param.split('=')
                try:
                    v = float(v)
                except ValueError:
                    pass
                dc[k] = v
            else:
                pass
        return dc

    @property
    def isGeo(self):
        if False:
            print('Hello World!')
        if self.code == 4326:
            return True
        elif HAS_GDAL:
            prj = self.getOgrSpatialRef()
            isGeo = prj.IsGeographic()
            return isGeo == 1
        elif HAS_PYPROJ:
            prj = self.getPyProj()
            return prj.crs.is_geographic
        else:
            return None

    def getWKT(self):
        if False:
            return 10
        if HAS_GDAL:
            prj = self.getOgrSpatialRef()
            return prj.ExportToWkt()
        elif self.isEPSG:
            return EPSGIO.getEsriWkt(self.code)
        else:
            raise NotImplementedError