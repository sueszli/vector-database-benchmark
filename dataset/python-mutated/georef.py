import math
from ..proj import SRS
from ..utils import XY as xy, BBOX
from ..errors import OverlapError

class GeoRef:
    """
	Represents georefencing informations of a raster image
	Note : image origin is upper-left whereas map origin is lower-left
	"""

    def __init__(self, rSize, pxSize, origin, rot=xy(0, 0), pxCenter=True, subBoxGeo=None, crs=None):
        if False:
            for i in range(10):
                print('nop')
        "\n\t\trSize : dimensions of the raster in pixels (width, height) tuple\n\t\tpxSize : dimension of a pixel in map units (x scale, y scale) tuple. y is always negative\n\t\torigin : upper left geo coords of pixel center, (x, y) tuple\n\t\tpxCenter : set it to True is the origin anchor point is located at pixel center\n\t\t\tor False if it's lolcated at pixel corner\n\t\trotation : rotation terms (xrot, yrot) <--> (yskew, xskew)\n\t\tsubBoxGeo : a BBOX object that define the working extent (subdataset) in geographic coordinate space\n\t\t"
        self.rSize = xy(*rSize)
        self.origin = xy(*origin)
        self.pxSize = xy(*pxSize)
        if not pxCenter:
            self.origin[0] += abs(self.pxSize.x / 2)
            self.origin[1] -= abs(self.pxSize.y / 2)
        self.rotation = xy(*rot)
        if subBoxGeo is not None:
            self.setSubBoxGeo(subBoxGeo)
        else:
            self.subBoxGeo = None
        if crs is not None:
            if isinstance(crs, SRS):
                self.crs = crs
            else:
                raise IOError('CRS must be SRS() class object not ' + str(type(crs)))
        else:
            self.crs = crs

    @classmethod
    def fromGDAL(cls, ds):
        if False:
            i = 10
            return i + 15
        'init from gdal dataset instance'
        geoTrans = ds.GetGeoTransform()
        if geoTrans is not None:
            (xmin, resx, rotx, ymax, roty, resy) = geoTrans
            (w, h) = (ds.RasterXSize, ds.RasterYSize)
            try:
                crs = SRS.fromGDAL(ds)
            except Exception as e:
                crs = None
            return cls((w, h), (resx, resy), (xmin, ymax), rot=(rotx, roty), pxCenter=False, crs=crs)
        else:
            return None

    @classmethod
    def fromWorldFile(cls, wfPath, rasterSize):
        if False:
            return 10
        'init from a worldfile'
        try:
            with open(wfPath, 'r') as f:
                wf = f.readlines()
            pxSize = xy(float(wf[0].replace(',', '.')), float(wf[3].replace(',', '.')))
            rotation = xy(float(wf[1].replace(',', '.')), float(wf[2].replace(',', '.')))
            origin = xy(float(wf[4].replace(',', '.')), float(wf[5].replace(',', '.')))
            return cls(rasterSize, pxSize, origin, rot=rotation, pxCenter=True, crs=None)
        except Exception as e:
            raise IOError('Unable to read worldfile. {}'.format(e))

    @classmethod
    def fromTyf(cls, tif):
        if False:
            print('Hello World!')
        'read geotags from Tyf instance'
        (w, h) = (tif['ImageWidth'], tif['ImageLength'])
        try:
            transfoMatrix = tif['ModelTransformationTag']
        except KeyError:
            transfoMatrix = None
        try:
            modelTiePoint = tif['ModelTiepointTag']
            modelPixelScale = tif['ModelPixelScaleTag']
        except KeyError:
            modelTiePoint = None
            modelPixelScale = None
        if transfoMatrix is not None:
            (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) = transfoMatrix
            origin = xy(d, h)
            pxSize = xy(a, f)
            rotation = xy(e, b)
        elif modelTiePoint is not None and modelPixelScale is not None:
            origin = xy(*modelTiePoint[3:5])
            pxSize = xy(*modelPixelScale[0:2])
            pxSize[1] = -pxSize.y
            rotation = xy(0, 0)
        else:
            raise IOError('Unable to read geotags')
        try:
            geotags = tif['GeoKeyDirectoryTag']
        except KeyError:
            cellAnchor = 1
        else:
            cellAnchor = geotags[geotags.index(1025) + 3]
        if cellAnchor == 1:
            origin[0] += abs(pxSize.x / 2)
            origin[1] -= abs(pxSize.y / 2)
        return cls((w, h), pxSize, origin, rot=rotation, pxCenter=True, crs=None)

    def toGDAL(self):
        if False:
            i = 10
            return i + 15
        'return a tuple of georef parameters ordered to define geotransformation of a gdal datasource'
        (xmin, ymax) = self.corners[0]
        (xres, yres) = self.pxSize
        (xrot, yrot) = self.rotation
        return (xmin, xres, xrot, ymax, yrot, yres)

    def toWorldFile(self, path):
        if False:
            i = 10
            return i + 15
        'export geo transformation to a worldfile'
        (xmin, ymax) = self.origin
        (xres, yres) = self.pxSize
        (xrot, yrot) = self.rotation
        wf = (xres, xrot, yrot, yres, xmin, ymax)
        f = open(path, 'w')
        f.write('\n'.join(list(map(str, wf))))
        f.close()

    @property
    def hasCRS(self):
        if False:
            return 10
        return self.crs is not None

    @property
    def hasRotation(self):
        if False:
            print('Hello World!')
        return self.rotation.x != 0 or self.rotation.y != 0
    "\n\t@property\n\tdef ul(self):\n\t\t'''upper left corner'''\n\t\treturn self.geoFromPx(0, yPxRange, True)\n\t@property\n\tdef ur(self):\n\t\t'''upper right corner'''\n\t\treturn self.geoFromPx(xPxRange, yPxRange, True)\n\t@property\n\tdef bl(self):\n\t\t'''bottom left corner'''\n\t\treturn self.geoFromPx(0, 0, True)\n\t@property\n\tdef br(self):\n\t\t'''bottom right corner'''\n\t\treturn self.geoFromPx(xPxRange, 0, True)\n\t"

    @property
    def cornersCenter(self):
        if False:
            print('Hello World!')
        '\n\t\t(x,y) geo coordinates of image corners (upper left, upper right, bottom right, bottom left)\n\t\t(pt1, pt2, pt3, pt4) <--> (upper left, upper right, bottom right, bottom left)\n\t\tThe coords are located at the pixel center\n\t\t'
        xPxRange = self.rSize.x - 1
        yPxRange = self.rSize.y - 1
        pt1 = self.geoFromPx(0, 0, pxCenter=True)
        pt2 = self.geoFromPx(xPxRange, 0, pxCenter=True)
        pt3 = self.geoFromPx(xPxRange, yPxRange, pxCenter=True)
        pt4 = self.geoFromPx(0, yPxRange, pxCenter=True)
        return (pt1, pt2, pt3, pt4)

    @property
    def corners(self):
        if False:
            while True:
                i = 10
        '\n\t\t(x,y) geo coordinates of image corners (upper left, upper right, bottom right, bottom left)\n\t\t(pt1, pt2, pt3, pt4) <--> (upper left, upper right, bottom right, bottom left)\n\t\tRepresent the true corner location (upper left for pt1, upper right for pt2 ...)\n\t\t'
        (pt1, pt2, pt3, pt4) = self.cornersCenter
        xOffset = abs(self.pxSize.x / 2)
        yOffset = abs(self.pxSize.y / 2)
        pt1 = xy(pt1.x - xOffset, pt1.y + yOffset)
        pt2 = xy(pt2.x + xOffset, pt2.y + yOffset)
        pt3 = xy(pt3.x + xOffset, pt3.y - yOffset)
        pt4 = xy(pt4.x - xOffset, pt4.y - yOffset)
        return (pt1, pt2, pt3, pt4)

    @property
    def bbox(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a bbox class object'
        pts = self.corners
        xmin = min([pt.x for pt in pts])
        xmax = max([pt.x for pt in pts])
        ymin = min([pt.y for pt in pts])
        ymax = max([pt.y for pt in pts])
        return BBOX(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @property
    def bboxPx(self):
        if False:
            for i in range(10):
                print('nop')
        return BBOX(xmin=0, ymin=0, xmax=self.rSize.x, ymax=self.rSize.y)

    @property
    def center(self):
        if False:
            i = 10
            return i + 15
        '(x,y) geo coordinates of image center'
        return xy(self.corners[0].x + self.geoSize.x / 2, self.corners[0].y - self.geoSize.y / 2)

    @property
    def geoSize(self):
        if False:
            print('Hello World!')
        'raster dimensions (width, height) in map units'
        return xy(self.rSize.x * abs(self.pxSize.x), self.rSize.y * abs(self.pxSize.y))

    @property
    def orthoGeoSize(self):
        if False:
            while True:
                i = 10
        'ortho geo size when affine transfo applied a rotation'
        pxWidth = math.sqrt(self.pxSize.x ** 2 + self.rotation.x ** 2)
        pxHeight = math.sqrt(self.pxSize.y ** 2 + self.rotation.y ** 2)
        return xy(self.rSize.x * pxWidth, self.rSize.y * pxHeight)

    @property
    def orthoPxSize(self):
        if False:
            for i in range(10):
                print('nop')
        'ortho pixels size when affine transfo applied a rotation'
        pxWidth = math.sqrt(self.pxSize.x ** 2 + self.rotation.x ** 2)
        pxHeight = math.sqrt(self.pxSize.y ** 2 + self.rotation.y ** 2)
        return xy(pxWidth, pxHeight)

    def geoFromPx(self, xPx, yPx, reverseY=False, pxCenter=True):
        if False:
            return 10
        '\n\t\tAffine transformation (cf. ESRI WorldFile spec.)\n\t\tReturn geo coords of the center of an given pixel\n\t\txPx = the column number of the pixel in the image counting from left\n\t\tyPx = the row number of the pixel in the image counting from top\n\t\tuse reverseY option is yPx is counting from bottom instead of top\n\t\tNumber of pixels is range from 0 (not 1)\n\t\t'
        if pxCenter:
            (xPx, yPx) = (math.floor(xPx), math.floor(yPx))
            (ox, oy) = (self.origin.x, self.origin.y)
        else:
            ox = self.origin.x - abs(self.pxSize.x / 2)
            oy = self.origin.y + abs(self.pxSize.y / 2)
        if reverseY:
            yPxRange = self.rSize.y - 1
            yPx = yPxRange - yPx
        x = self.pxSize.x * xPx + self.rotation.y * yPx + ox
        y = self.pxSize.y * yPx + self.rotation.x * xPx + oy
        return xy(x, y)

    def pxFromGeo(self, x, y, reverseY=False, round2Floor=False):
        if False:
            print('Hello World!')
        '\n\t\tAffine transformation (cf. ESRI WorldFile spec.)\n\t\tReturn pixel position of given geographic coords\n\t\tuse reverseY option to get y pixels counting from bottom\n\t\tPixels position is range from 0 (not 1)\n\t\t'
        (pxSizex, pxSizey) = self.pxSize
        (rotx, roty) = self.rotation
        offx = self.origin.x - abs(self.pxSize.x / 2)
        offy = self.origin.y + abs(self.pxSize.y / 2)
        xPx = (pxSizey * x - rotx * y + rotx * offy - pxSizey * offx) / (pxSizex * pxSizey - rotx * roty)
        yPx = (-roty * x + pxSizex * y + roty * offx - pxSizex * offy) / (pxSizex * pxSizey - rotx * roty)
        if reverseY:
            yPxRange = self.rSize.y - 1
            yPx = yPxRange - yPx
            yPx += 1
        if round2Floor:
            (xPx, yPx) = (math.floor(xPx), math.floor(yPx))
        return xy(xPx, yPx)

    def pxToGeo(self, xPx, yPx, reverseY=False):
        if False:
            for i in range(10):
                print('nop')
        return self.geoFromPx(xPx, yPx, reverseY)

    def geoToPx(self, x, y, reverseY=False, round2Floor=False):
        if False:
            for i in range(10):
                print('nop')
        return self.pxFromGeo(x, y, reverseY, round2Floor)

    def setSubBoxGeo(self, subBoxGeo):
        if False:
            while True:
                i = 10
        'set a subbox in geographic coordinate space\n\t\tif needed, coords will be adjusted to avoid being outside raster size'
        if self.hasRotation:
            raise IOError('A subbox cannot be define if the raster has rotation parameter')
        if not self.bbox.overlap(subBoxGeo):
            raise OverlapError()
        elif self.bbox.isWithin(subBoxGeo):
            return
        else:
            (xminPx, ymaxPx) = self.pxFromGeo(subBoxGeo.xmin, subBoxGeo.ymin, round2Floor=True)
            (xmaxPx, yminPx) = self.pxFromGeo(subBoxGeo.xmax, subBoxGeo.ymax, round2Floor=True)
            subBoxPx = BBOX(xmin=xminPx, ymin=yminPx, xmax=xmaxPx, ymax=ymaxPx)
            self.setSubBoxPx(subBoxPx)

    def setSubBoxPx(self, subBoxPx):
        if False:
            for i in range(10):
                print('nop')
        if not self.bboxPx.overlap(subBoxPx):
            raise OverlapError()
        (xminPx, xmaxPx) = (subBoxPx.xmin, subBoxPx.xmax)
        (yminPx, ymaxPx) = (subBoxPx.ymin, subBoxPx.ymax)
        (sizex, sizey) = self.rSize
        if xminPx < 0:
            xminPx = 0
        if xmaxPx >= sizex:
            xmaxPx = sizex - 1
        if yminPx < 0:
            yminPx = 0
        if ymaxPx >= sizey:
            ymaxPx = sizey - 1
        (xmin, ymin) = self.geoFromPx(xminPx, ymaxPx)
        (xmax, ymax) = self.geoFromPx(xmaxPx, yminPx)
        self.subBoxGeo = BBOX(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def applySubBox(self):
        if False:
            for i in range(10):
                print('nop')
        if self.subBoxGeo is not None:
            self.rSize = self.subBoxPxSize
            self.origin = self.subBoxGeoOrigin
            self.subBoxGeo = None

    def getSubBoxGeoRef(self):
        if False:
            print('Hello World!')
        return GeoRef(self.subBoxPxSize, self.pxSize, self.subBoxGeoOrigin, pxCenter=True, crs=self.crs)

    @property
    def subBoxPx(self):
        if False:
            for i in range(10):
                print('nop')
        'return the subbox as bbox object in pixels coordinates space'
        if self.subBoxGeo is None:
            return None
        (xmin, ymax) = self.pxFromGeo(self.subBoxGeo.xmin, self.subBoxGeo.ymin, round2Floor=True)
        (xmax, ymin) = self.pxFromGeo(self.subBoxGeo.xmax, self.subBoxGeo.ymax, round2Floor=True)
        return BBOX(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @property
    def subBoxPxSize(self):
        if False:
            while True:
                i = 10
        'dimension of the subbox in pixels'
        if self.subBoxGeo is None:
            return None
        bbpx = self.subBoxPx
        (w, h) = (bbpx.xmax - bbpx.xmin, bbpx.ymax - bbpx.ymin)
        return xy(w + 1, h + 1)

    @property
    def subBoxGeoSize(self):
        if False:
            print('Hello World!')
        'dimension of the subbox in map units'
        if self.subBoxGeo is None:
            return None
        (sizex, sizey) = self.subBoxPxSize
        return xy(sizex * abs(self.pxSize.x), sizey * abs(self.pxSize.y))

    @property
    def subBoxPxOrigin(self):
        if False:
            i = 10
            return i + 15
        'pixel coordinate of subbox origin'
        if self.subBoxGeo is None:
            return None
        return xy(self.subBoxPx.xmin, self.subBoxPx.ymin)

    @property
    def subBoxGeoOrigin(self):
        if False:
            for i in range(10):
                print('nop')
        'geo coordinate of subbox origin, adjusted at pixel center'
        if self.subBoxGeo is None:
            return None
        return xy(self.subBoxGeo.xmin, self.subBoxGeo.ymax)

    def __repr__(self):
        if False:
            return 10
        s = [' spatial ref system {}'.format(self.crs), ' origin geo {}'.format(self.origin), ' pixel size {}'.format(self.pxSize), ' rotation {}'.format(self.rotation), ' bounding box {}'.format(self.bbox), ' geoSize {}'.format(self.geoSize)]
        if self.subBoxGeo is not None:
            s.extend([' subbox origin (geo space) {}'.format(self.subBoxGeoOrigin), ' subbox origin (px space) {}'.format(self.subBoxPxOrigin), ' subbox (geo space) {}'.format(self.subBoxGeo), ' subbox (px space) {}'.format(self.subBoxPx), ' sub geoSize {}'.format(self.subBoxGeoSize), ' sub pxSize {}'.format(self.subBoxPxSize)])
        return '\n'.join(s)