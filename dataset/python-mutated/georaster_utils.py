import os
import numpy as np
import bpy, bmesh
import math
import logging
log = logging.getLogger(__name__)
from ...core.georaster import GeoRaster

def _exportAsMesh(georaster, dx=0, dy=0, step=1, buildFaces=True, flat=False, subset=False, reproj=None):
    if False:
        return 10
    'Numpy test'
    if subset and georaster.subBoxGeo is None:
        subset = False
    if not subset:
        georef = georaster.georef
    else:
        georef = georaster.getSubBoxGeoRef()
    (x0, y0) = georef.origin
    (pxSizeX, pxSizeY) = (georef.pxSize.x, georef.pxSize.y)
    (w, h) = (georef.rSize.x, georef.rSize.y)
    (w, h) = (math.ceil(w / step), math.ceil(h / step))
    (pxSizeX, pxSizeY) = (pxSizeX * step, pxSizeY * step)
    x = np.array([x0 + pxSizeX * i - dx for i in range(0, w)])
    y = np.array([y0 + pxSizeY * i - dy for i in range(0, h)])
    (xx, yy) = np.meshgrid(x, y)
    if flat:
        zz = np.zeros((h, w))
    else:
        zz = georaster.readAsNpArray(subset=subset).data[::step, ::step]
    verts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    if buildFaces:
        faces = [(x + y * w, x + y * w + 1, x + y * w + 1 + w, x + y * w + w) for x in range(0, w - 1) for y in range(0, h - 1)]
    else:
        faces = []
    mesh = bpy.data.meshes.new('DEM')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh

def exportAsMesh(georaster, dx=0, dy=0, step=1, buildFaces=True, subset=False, reproj=None, flat=False):
    if False:
        while True:
            i = 10
    if subset and georaster.subBoxGeo is None:
        subset = False
    if not subset:
        georef = georaster.georef
    else:
        georef = georaster.getSubBoxGeoRef()
    if not flat:
        img = georaster.readAsNpArray(subset=subset)
        data = img.data
    (x0, y0) = georef.origin
    (pxSizeX, pxSizeY) = (georef.pxSize.x, georef.pxSize.y)
    (w, h) = (georef.rSize.x, georef.rSize.y)
    verts = []
    faces = []
    nodata = []
    idxMap = {}
    for py in range(0, h, step):
        for px in range(0, w, step):
            x = x0 + pxSizeX * px
            y = y0 + pxSizeY * py
            if reproj is not None:
                (x, y) = reproj.pt(x, y)
            x -= dx
            y -= dy
            if flat:
                z = 0
            else:
                z = data[py, px]
            v1 = px + py * w
            if z == georaster.noData:
                nodata.append(v1)
            else:
                verts.append((x, y, z))
                idxMap[v1] = len(verts) - 1
                if buildFaces and px > 0 and (py > 0):
                    v2 = v1 - step
                    v3 = v2 - w * step
                    v4 = v3 + step
                    f = [v4, v3, v2, v1]
                    if not any((v in f for v in nodata)):
                        f = [idxMap[v] for v in f]
                        faces.append(f)
    mesh = bpy.data.meshes.new('DEM')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh

def rasterExtentToMesh(name, rast, dx, dy, pxLoc='CORNER', reproj=None, subdivise=False):
    if False:
        for i in range(10):
            print('nop')
    'Build a new mesh that represent a georaster extent'
    bm = bmesh.new()
    if pxLoc == 'CORNER':
        pts = [(pt[0], pt[1]) for pt in rast.corners]
    elif pxLoc == 'CENTER':
        pts = [(pt[0], pt[1]) for pt in rast.cornersCenter]
    if reproj is not None:
        pts = reproj.pts(pts)
    pts = [bm.verts.new((pt[0] - dx, pt[1] - dy, 0)) for pt in pts]
    pts.reverse()
    bm.faces.new(pts)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh

def geoRastUVmap(obj, uvLayer, rast, dx, dy, reproj=None):
    if False:
        print('Hello World!')
    'uv map a georaster texture on a given mesh'
    mesh = obj.data
    loc = obj.location
    for pg in mesh.polygons:
        for i in pg.loop_indices:
            vertIdx = mesh.loops[i].vertex_index
            pt = list(mesh.vertices[vertIdx].co)
            pt = (pt[0] + loc.x + dx, pt[1] + loc.y + dy)
            if reproj is not None:
                pt = reproj.pt(*pt)
            (dx_px, dy_px) = rast.pxFromGeo(pt[0], pt[1], reverseY=True, round2Floor=False)
            u = dx_px / rast.size[0]
            v = dy_px / rast.size[1]
            uvLayer.data[i].uv = [u, v]

def setDisplacer(obj, rast, uvTxtLayer, mid=0, interpolation=False):
    if False:
        while True:
            i = 10
    displacer = obj.modifiers.new('DEM', type='DISPLACE')
    demTex = bpy.data.textures.new('demText', type='IMAGE')
    demTex.image = rast.bpyImg
    demTex.use_interpolation = interpolation
    demTex.extension = 'CLIP'
    demTex.use_clamp = False
    displacer.texture = demTex
    displacer.texture_coords = 'UV'
    displacer.uv_layer = uvTxtLayer.name
    displacer.mid_level = mid
    if rast.depth < 32:
        displacer.strength = 2 ** rast.depth - 1
    else:
        displacer.strength = 1
    bpy.ops.object.shade_smooth()
    return displacer

class bpyGeoRaster(GeoRaster):

    def __init__(self, path, subBoxGeo=None, useGDAL=False, clip=False, fillNodata=False, raw=False):
        if False:
            while True:
                i = 10
        GeoRaster.__init__(self, path, subBoxGeo=subBoxGeo, useGDAL=useGDAL)
        if self.format not in ['GTiff', 'TIFF', 'BMP', 'PNG', 'JPEG', 'JPEG2000'] or (clip and self.subBoxGeo is not None) or fillNodata or (self.ddtype == 'int16'):
            if clip:
                img = self.readAsNpArray(subset=True)
            else:
                img = self.readAsNpArray()
            img.cast2float()
            if fillNodata:
                img.fillNodata()
            filepath = os.path.splitext(self.path)[0] + '_bgis.tif'
            img.save(filepath)
            GeoRaster.__init__(self, filepath, useGDAL=useGDAL)
        self.raw = raw
        self._load()

    def _load(self, pack=False):
        if False:
            return 10
        'Load the georaster in Blender'
        try:
            self.bpyImg = bpy.data.images.load(self.path)
        except Exception as e:
            log.error('Unable to open raster', exc_info=True)
            raise IOError('Unable to open raster')
        if pack:
            self.bpyImg.pack()
        if self.raw:
            self.bpyImg.colorspace_settings.is_data = True

    def unload(self):
        if False:
            print('Hello World!')
        self.bpyImg.user_clear()
        bpy.data.images.remove(self.bpyImg)
        self.bpyImg = None

    @property
    def isLoaded(self):
        if False:
            while True:
                i = 10
        'Flag if the image has been loaded in Blender'
        if self.bpyImg is not None:
            return True
        else:
            return False

    @property
    def isPacked(self):
        if False:
            i = 10
            return i + 15
        'Flag if the image has been packed in Blender'
        if self.bpyImg is not None:
            if len(self.bpyImg.packed_files) == 0:
                return False
            else:
                return True
        else:
            return False

    def toBitDepth(self, a):
        if False:
            while True:
                i = 10
        '\n\t\tConvert Blender pixel intensity value (from 0.0 to 1.0)\n\t\tin true pixel value in initial image bit depth range\n\t\t'
        return a * (2 ** self.depth - 1)

    def fromBitDepth(self, a):
        if False:
            print('Hello World!')
        '\n\t\tConvert true pixel value in initial image bit depth range\n\t\tto Blender pixel intensity value (from 0.0 to 1.0)\n\t\t'
        return a / (2 ** self.depth - 1)

    def getPixelsArray(self, bandIdx=None, subset=False):
        if False:
            i = 10
            return i + 15
        "\n\t\tUse bpy to extract pixels values as numpy array\n\t\tIn numpy fist dimension of a 2D matrix represents rows (y) and second dimension represents cols (x)\n\t\tso to get pixel value at a specified location be careful not confusing axes: data[row, column]\n\t\tIt's possible to swap axes if you prefere accessing values with [x,y] indices instead of [y,x]: data.swapaxes(0,1)\n\t\tArray origin is top left\n\t\t"
        if not self.isLoaded:
            raise IOError('Can read only image opened in Blender')
        if self.ddtype is None:
            raise IOError('Undefined data type')
        if subset and self.subBoxGeo is None:
            return None
        nbBands = self.bpyImg.channels
        a = np.array(self.bpyImg.pixels[:])
        a = a.reshape(len(a) / nbBands, nbBands)
        a = a.reshape(self.size.y, self.size.x, nbBands)
        a = np.flipud(a)
        if bandIdx is not None:
            a = a[:, :, bandIdx]
        if not self.isFloat:
            a = self.toBitDepth(a)
            a = np.rint(a).astype(self.ddtype)
            "\n\t\t\tif self.ddtype == 'int16':\n\t\t\t\t#16 bits allows coding values from 0 to 65535 (with 65535 == 2**depth / 2 - 1 )\n\t\t\t\t#positives value are coded from 0 to 32767 (from 0.0 to 0.5 in Blender)\n\t\t\t\t#negatives values are coded in reverse order from 65535 to 32768 (1.0 to 0.5 in Blender)\n\t\t\t\t#corresponding to a range from -1 to -32768\n\t\t\t\ta = np.where(a > 32767, -(65536-a), a)\n\t\t\t"
        if not subset:
            return a
        else:
            subBoxPx = self.subBoxPx
            a = a[subBoxPx.ymin:subBoxPx.ymax + 1, subBoxPx.xmin:subBoxPx.xmax + 1]
            return a

    def flattenPixelsArray(self, px):
        if False:
            print('Hello World!')
        '\n\t\tFlatten a 3d array of pixels to match the shape of bpy.pixels\n\t\t[ [[rgba], [rgba]...], [lines2], [lines3]...] >> [r,g,b,a,r,g,b,a,r,g,b,a, ... ]\n\t\tIf the submited array contains only one band, then the band will be duplicate\n\t\tand an alpha band will be added to get all rgba values.\n\t\t'
        shape = px.shape
        if len(shape) == 2:
            px = np.expand_dims(px, axis=2)
            px = np.repeat(px, 3, axis=2)
            alpha = np.ones(shape)
            alpha = np.expand_dims(alpha, axis=2)
            px = np.append(px, alpha, axis=2)
        px = np.flipud(px)
        px = px.flatten()
        return px