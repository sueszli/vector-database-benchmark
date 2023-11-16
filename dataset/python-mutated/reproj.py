import math
from .srs import SRS
from .utm import UTM, UTM_EPSG_CODES
from .ellps import GRS80
from .srv import EPSGIO
from ..errors import ReprojError
from ..utils import BBOX
from ..checkdeps import HAS_GDAL, HAS_PYPROJ
from .. import settings
if HAS_GDAL:
    from osgeo import osr, gdal
if HAS_PYPROJ:
    import pyproj

def webMercToLonLat(x, y):
    if False:
        while True:
            i = 10
    k = GRS80.perimeter / 360
    lon = x / k
    lat = y / k
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return (lon, lat)

def lonLatToWebMerc(lon, lat):
    if False:
        print('Hello World!')
    k = GRS80.perimeter / 360
    x = lon * k
    lat = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = lat * k
    return (x, y)

def reprojImg(crs1, crs2, ds1, out_ul=None, out_size=None, out_res=None, sqPx=False, resamplAlg='BL', path=None, geoTiffOptions={'TFW': 'YES', 'TILED': 'YES', 'BIGTIFF': 'YES', 'COMPRESS': 'JPEG', 'JPEG_QUALITY': 80, 'PHOTOMETRIC': 'YCBCR'}):
    if False:
        return 10
    '\n\tUse GDAL Python binding to reproject an image\n\tcrs1, crs2 >> epsg code\n\tds1 >> input GDAL dataset object\n\tout_ul >> [tuple] output raster top left coords (same as input if None)\n\tout_size >> |tuple], output raster size (same as input is None)\n\tout_res >> [number], output raster resolution (same as input if None) (resx = resy)\n\tsqPx >> [boolean] force square pixel resolution when resoltion is automatically computed\n\tpath >> a geotiff file path to store the result into (optional)\n\tgeoTiffOptions >> GDAL create option for tiff format (optional)\n\treturn ds2 >> output GDAL dataset object. If path is None, the dataset will be stored in memory however into a geotiff file on disk\n\t'
    if not HAS_GDAL:
        raise NotImplementedError
    geoTrans = ds1.GetGeoTransform()
    if geoTrans is not None:
        (xmin, resx, rotx, ymax, roty, resy) = geoTrans
    else:
        raise IOError('Reprojection fails: input raster is not georeferenced')
    (img_w, img_h) = (ds1.RasterXSize, ds1.RasterYSize)
    nbBands = ds1.RasterCount
    dtype = gdal.GetDataTypeName(ds1.GetRasterBand(1).DataType)
    if rotx == roty == 0:
        xmax = xmin + img_w * resx
        ymin = ymax + img_h * resy
        bbox = BBOX(xmin, ymin, xmax, ymax)
    else:
        raise IOError('Raster must be rectified (no rotation parameters)')
    prj1 = SRS(crs1).getOgrSpatialRef()
    wkt1 = prj1.ExportToWkt()
    ds1.SetProjection(wkt1)
    if out_ul is not None:
        (xmin, ymax) = out_ul
    else:
        (xmin, ymax) = reprojPt(crs1, crs2, xmin, ymax)
    if out_res is not None and out_size is not None:
        (resx, resy) = (out_res, -out_res)
        (img_w, img_h) = out_size
    if out_res is not None and out_size is None:
        (resx, resy) = (out_res, -out_res)
        (xmin, ymin, xmax, ymax) = reprojBbox(crs1, crs2, bbox)
        img_w = int((xmax - xmin) / resx)
        img_h = int((ymax - ymin) / resy)
    if out_res is None and out_size is not None:
        (img_w, img_h) = out_size
    if out_res is None and out_size is None:
        (xmin, ymin, xmax, ymax) = reprojBbox(crs1, crs2, bbox)
        '\n\t\tdst_diag = math.sqrt( (xmax - xmin)**2 + (ymax - ymin)**2)\n\t\tpx_diag = math.sqrt(img_w**2 + img_h**2)\n\t\tres = dst_diag / px_diag\n\t\t'
        resx = (xmax - xmin) / img_w
        resy = -(ymax - ymin) / img_h
        if sqPx:
            resx = max(resx, abs(resy))
            resy = -resx
    if path is None:
        ds2 = gdal.GetDriverByName('MEM').Create('', img_w, img_h, nbBands, gdal.GetDataTypeByName(dtype))
    else:
        gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
        options = [str(k) + '=' + str(v) for (k, v) in geoTiffOptions.items()]
        ds2 = gdal.GetDriverByName('GTiff').Create(path, img_w, img_h, nbBands, gdal.GetDataTypeByName(dtype), options)
        if geoTiffOptions.get('COMPRESS', None) == 'JPEG':
            ds2.CreateMaskBand(gdal.GMF_PER_DATASET)
            ds2.GetRasterBand(1).GetMaskBand().Fill(255)
    geoTrans = (xmin, resx, 0, ymax, 0, resy)
    ds2.SetGeoTransform(geoTrans)
    prj2 = SRS(crs2).getOgrSpatialRef()
    wkt2 = prj2.ExportToWkt()
    ds2.SetProjection(wkt2)
    if resamplAlg == 'NN':
        alg = gdal.GRA_NearestNeighbour
    elif resamplAlg == 'BL':
        alg = gdal.GRA_Bilinear
    elif resamplAlg == 'CB':
        alg = gdal.GRA_Cubic
    elif resamplAlg == 'CBS':
        alg = gdal.GRA_CubicSpline
    elif resamplAlg == 'LCZ':
        alg = gdal.GRA_Lanczos
    memLimit = 0
    threshold = 0.25
    opt = ['NUM_THREADS=ALL_CPUS, SAMPLE_GRID=YES']
    (a, b, c) = gdal.__version__.split('.', 2)
    if int(a) == 2 and int(b) >= 1 or int(a) > 2:
        gdal.ReprojectImage(ds1, ds2, wkt1, wkt2, alg, memLimit, threshold, options=opt)
    else:
        gdal.ReprojectImage(ds1, ds2, wkt1, wkt2, alg, memLimit, threshold)
    return ds2

class Reproj:

    def __init__(self, crs1, crs2):
        if False:
            while True:
                i = 10
        try:
            (crs1, crs2) = (SRS(crs1), SRS(crs2))
        except Exception as e:
            raise ReprojError(str(e))
        if crs1 == crs2:
            self.iproj = 'NO_REPROJ'
            return
        self.iproj = settings.proj_engine
        if self.iproj not in ['AUTO', 'GDAL', 'PYPROJ', 'BUILTIN', 'EPSGIO']:
            raise ReprojError('Wrong engine name')
        if self.iproj == 'AUTO':
            if HAS_GDAL:
                self.iproj = 'GDAL'
            elif HAS_PYPROJ:
                self.iproj = 'PYPROJ'
            elif (crs1.isWM or crs1.isUTM) and crs2.isWGS84 or (crs1.isWGS84 and (crs2.isWM or crs2.isUTM)):
                self.iproj = 'BUILTIN'
            elif EPSGIO.ping():
                self.iproj = 'EPSGIO'
            else:
                raise ReprojError('Too limited reprojection capabilities.')
        else:
            if self.iproj == 'GDAL' and (not HAS_GDAL) or (self.iproj == 'PYPROJ' and (not HAS_PYPROJ)):
                raise ReprojError('Missing reproj engine')
            if self.iproj == 'BUILTIN':
                if not ((crs1.isWM or crs1.isUTM) and crs2.isWGS84 or (crs1.isWGS84 and (crs2.isWM or crs2.isUTM))):
                    raise ReprojError('Too limited built in reprojection capabilities')
            if self.iproj == 'EPSGIO':
                if not EPSGIO.ping():
                    raise ReprojError('Cannot access epsg.io service')
        if self.iproj == 'GDAL':
            self.crs1 = crs1.getOgrSpatialRef()
            self.crs2 = crs2.getOgrSpatialRef()
            self.osrTransfo = osr.CoordinateTransformation(self.crs1, self.crs2)
        elif self.iproj == 'PYPROJ':
            self.crs1 = crs1.getPyProj()
            self.crs2 = crs2.getPyProj()
        elif self.iproj == 'EPSGIO':
            if crs1.isEPSG and crs2.isEPSG:
                (self.crs1, self.crs2) = (crs1.code, crs2.code)
            else:
                raise ReprojError('EPSG.io support only EPSG code')
        elif self.iproj == 'BUILTIN':
            if (crs1.isWM or crs1.isUTM) and crs2.isWGS84 or (crs1.isWGS84 and (crs2.isWM or crs2.isUTM)):
                (self.crs1, self.crs2) = (crs1.code, crs2.code)
            else:
                raise ReprojError('Not implemented transformation')
            if crs1.isUTM:
                self.utm = UTM.init_from_epsg(crs1)
            elif crs2.isUTM:
                self.utm = UTM.init_from_epsg(crs2)

    def pts(self, pts):
        if False:
            print('Hello World!')
        if len(pts) == 0:
            return []
        if len(pts[0]) != 2:
            raise ReprojError('Points must be [ (x,y) ]')
        if self.iproj == 'NO_REPROJ':
            return pts
        if self.iproj == 'GDAL':
            if hasattr(osr, 'GetPROJVersionMajor'):
                projVersion = osr.GetPROJVersionMajor()
            else:
                projVersion = 4
            if projVersion >= 6 and self.crs1.IsGeographic():
                pts = [(pt[1], pt[0]) for pt in pts]
            if self.crs2.IsGeographic():
                (ys, xs, _zs) = zip(*self.osrTransfo.TransformPoints(pts))
            else:
                (xs, ys, _zs) = zip(*self.osrTransfo.TransformPoints(pts))
            return list(zip(xs, ys))
        elif self.iproj == 'PYPROJ':
            if self.crs1.crs.is_geographic:
                (ys, xs) = zip(*pts)
            else:
                (xs, ys) = zip(*pts)
            transformer = pyproj.Transformer.from_proj(self.crs1, self.crs2)
            if self.crs2.crs.is_geographic:
                (ys, xs) = transformer.transform(xs, ys)
            else:
                (xs, ys) = transformer.transform(xs, ys)
            return list(zip(xs, ys))
        elif self.iproj == 'EPSGIO':
            return EPSGIO.reprojPts(self.crs1, self.crs2, pts)
        elif self.iproj == 'BUILTIN':
            if self.crs1 == 4326 and self.crs2 == 3857:
                return [lonLatToWebMerc(*pt) for pt in pts]
            elif self.crs1 == 3857 and self.crs2 == 4326:
                return [webMercToLonLat(*pt) for pt in pts]
            if self.crs1 == 4326 and self.crs2 in UTM_EPSG_CODES:
                return [self.utm.lonlat_to_utm(*pt) for pt in pts]
            elif self.crs1 in UTM_EPSG_CODES and self.crs2 == 4326:
                return [self.utm.utm_to_lonlat(*pt) for pt in pts]

    def pt(self, x, y):
        if False:
            while True:
                i = 10
        if x is None or y is None:
            raise ReprojError('Cannot reproj None coordinates')
        return self.pts([(x, y)])[0]

    def bbox(self, bbox):
        if False:
            while True:
                i = 10
        'io type = BBOX() class'
        if not isinstance(bbox, BBOX):
            bbox = BBOX(*bbox)
        corners = self.pts(bbox.corners)
        _xmin = min((pt[0] for pt in corners))
        _xmax = max((pt[0] for pt in corners))
        _ymin = min((pt[1] for pt in corners))
        _ymax = max((pt[1] for pt in corners))
        if bbox.hasZ:
            return BBOX(_xmin, _ymin, bbox.zmin, _xmax, _ymax, bbox.zmax)
        else:
            return BBOX(_xmin, _ymin, _xmax, _ymax)

def reprojPt(crs1, crs2, x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n\tReproject x1,y1 coords from crs1 to crs2\n\tcrs can be an EPSG code (interger or string) or a proj4 string\n\tWARN : do not use this function in a loop because Reproj() init is slow\n\t'
    rprj = Reproj(crs1, crs2)
    return rprj.pt(x, y)

def reprojPts(crs1, crs2, pts):
    if False:
        print('Hello World!')
    '\n\tReproject [pts] from crs1 to crs2\n\tcrs can be an EPSG code (integer or srid string) or a proj4 string\n\tpts must be [(x,y)]\n\tWARN : do not use this function in a loop because Reproj() init is slow\n\t'
    rprj = Reproj(crs1, crs2)
    return rprj.pts(pts)

def reprojBbox(crs1, crs2, bbox):
    if False:
        return 10
    rprj = Reproj(crs1, crs2)
    return rprj.bbox(bbox)