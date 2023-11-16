import re
import os
import string
import bpy
import math
import string
import logging
log = logging.getLogger(__name__)
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator
from ..core.proj import Reproj
from ..core.utils import XY
from ..geoscene import GeoScene, georefManagerLayout
from ..prefs import PredefCRS
from .utils import bpyGeoRaster as GeoRaster
from .utils import placeObj, adjust3Dview, showTextures, addTexture, getBBOX
from .utils import rasterExtentToMesh, geoRastUVmap, setDisplacer
(PKG, SUBPKG) = __package__.split('.', maxsplit=1)

class IMPORTGIS_OT_ascii_grid(Operator, ImportHelper):
    """Import ESRI ASCII grid file"""
    bl_idname = 'importgis.asc_file'
    bl_description = 'Import ESRI ASCII grid with world file'
    bl_label = 'Import ASCII Grid'
    bl_options = {'UNDO'}
    filter_glob: StringProperty(default='*.asc;*.grd', options={'HIDDEN'})

    def listPredefCRS(self, context):
        if False:
            while True:
                i = 10
        return PredefCRS.getEnumItems()
    fileCRS: EnumProperty(name='CRS', description='Choose a Coordinate Reference System', items=listPredefCRS)
    importMode: EnumProperty(name='Mode', description='Select import mode', items=[('MESH', 'Mesh', 'Create triangulated regular network mesh'), ('CLOUD', 'Point cloud', 'Create vertex point cloud')])
    step: IntProperty(name='Step', description='Only read every Nth point for massive point clouds', default=1, min=1)
    newlines: BoolProperty(name='Newline-delimited rows', description='Use this method if the file contains newline separated rows for faster import', default=True)

    def draw(self, context):
        if False:
            return 10
        layout = self.layout
        layout.prop(self, 'importMode')
        layout.prop(self, 'step')
        layout.prop(self, 'newlines')
        row = layout.row(align=True)
        split = row.split(factor=0.35, align=True)
        split.label(text='CRS:')
        split.prop(self, 'fileCRS', text='')
        row.operator('bgis.add_predef_crs', text='', icon='ADD')
        scn = bpy.context.scene
        geoscn = GeoScene(scn)
        if geoscn.isPartiallyGeoref:
            georefManagerLayout(self, context)

    def total_lines(self, filename):
        if False:
            return 10
        '\n        Count newlines in file.\n        512MB file ~3 seconds.\n        '
        with open(filename) as f:
            lines = 0
            for _ in f:
                lines += 1
            return lines

    def read_row_newlines(self, f, ncols):
        if False:
            print('Hello World!')
        '\n        Read a row by columns separated by newline.\n        '
        return f.readline().split()

    def read_row_whitespace(self, f, ncols):
        if False:
            print('Hello World!')
        '\n        Read a row by columns separated by whitespace (including newlines).\n        6x slower than readlines() method but faster than any other method I can come up with. See commit 4d337c4 for alternatives.\n        '
        buf_size = min(1024 * 32, ncols * 6)
        row = []
        read_f = f.read
        while True:
            chunk = read_f(buf_size)
            if len(chunk) == buf_size:
                for i in range(len(chunk) - 1, -1, -1):
                    if chunk[i].isspace():
                        f.seek(f.tell() - (len(chunk) - i))
                        chunk = chunk[:i]
                        break
            if not chunk:
                return row
            for m in re.finditer('([^\\s]+)', chunk):
                row.append(m.group(0))
                if len(row) == ncols:
                    f.seek(f.tell() - (len(chunk) - m.end()))
                    return row

    @classmethod
    def poll(cls, context):
        if False:
            print('Hello World!')
        return context.mode == 'OBJECT'

    def execute(self, context):
        if False:
            print('Hello World!')
        prefs = context.preferences.addons[PKG].preferences
        bpy.ops.object.select_all(action='DESELECT')
        scn = bpy.context.scene
        geoscn = GeoScene(scn)
        if geoscn.isBroken:
            self.report({'ERROR'}, 'Scene georef is broken, please fix it beforehand')
            return {'CANCELLED'}
        if geoscn.isGeoref:
            (dx, dy) = geoscn.getOriginPrj()
        scale = geoscn.scale
        if not geoscn.hasCRS:
            try:
                geoscn.crs = self.fileCRS
            except Exception as e:
                log.error('Cannot set scene crs', exc_info=True)
                self.report({'ERROR'}, 'Cannot set scene crs, check logs for more infos')
                return {'CANCELLED'}
        if geoscn.crs != self.fileCRS:
            rprj = True
            rprjToRaster = Reproj(geoscn.crs, self.fileCRS)
            rprjToScene = Reproj(self.fileCRS, geoscn.crs)
        else:
            rprj = False
            rprjToRaster = None
            rprjToScene = None
        filename = self.filepath
        name = os.path.splitext(os.path.basename(filename))[0]
        log.info('Importing {}...'.format(filename))
        f = open(filename, 'r')
        meta_re = re.compile('^([^\\s]+)\\s+([^\\s]+)$')
        meta = {}
        for i in range(6):
            line = f.readline()
            m = meta_re.match(line)
            if m:
                meta[m.group(1).lower()] = m.group(2)
        log.debug(meta)
        step = self.step
        nrows = int(meta['nrows'])
        ncols = int(meta['ncols'])
        cellsize = float(meta['cellsize'])
        nodata = float(meta['nodata_value'])
        reprojection = {}
        offset = XY(0, 0)
        if 'xllcorner' in meta:
            llcorner = XY(float(meta['xllcorner']), float(meta['yllcorner']))
            reprojection['from'] = llcorner
        elif 'xllcenter' in meta:
            centre = XY(float(meta['xllcenter']), float(meta['yllcenter']))
            offset = XY(-cellsize / 2, -cellsize / 2)
            reprojection['from'] = centre
        if rprj:
            reprojection['to'] = XY(*rprjToScene.pt(*reprojection['from']))
            log.debug('{name} reprojected from {from} to {to}'.format(**reprojection, name=name))
        else:
            reprojection['to'] = reprojection['from']
        if not geoscn.isGeoref:
            centre = (reprojection['from'].x + offset.x + ncols / 2 * cellsize, reprojection['from'].y + offset.y + nrows / 2 * cellsize)
            if rprj:
                centre = rprjToScene.pt(*centre)
            geoscn.setOriginPrj(*centre)
            (dx, dy) = geoscn.getOriginPrj()
        index = 0
        vertices = []
        faces = []
        read = self.read_row_whitespace
        if self.newlines:
            read = self.read_row_newlines
        for y in range(nrows - 1, -1, -step):
            coldata = read(f, ncols)
            if len(coldata) != ncols:
                log.error('Incorrect number of columns for row {row}. Expected {expected}, got {actual}.'.format(row=nrows - y, expected=ncols, actual=len(coldata)))
                self.report({'ERROR'}, 'Incorrect number of columns for row, check logs for more infos')
                return {'CANCELLED'}
            for i in range(step - 1):
                _ = read(f, ncols)
            for x in range(0, ncols, step):
                if not (self.importMode == 'CLOUD' and coldata[x] == nodata):
                    pt = (x * cellsize + offset.x, y * cellsize + offset.y)
                    if rprj:
                        pt = rprjToScene.pt(pt[0] + reprojection['from'].x, pt[1] + reprojection['from'].y)
                        pt = (pt[0] - reprojection['to'].x, pt[1] - reprojection['to'].y)
                    try:
                        vertices.append(pt + (float(coldata[x]),))
                    except ValueError as e:
                        log.error('Value "{val}" in row {row}, column {col} could not be converted to a float.'.format(val=coldata[x], row=nrows - y, col=x))
                        self.report({'ERROR'}, 'Cannot convert value to float')
                        return {'CANCELLED'}
        if self.importMode == 'MESH':
            step_ncols = math.ceil(ncols / step)
            for r in range(0, math.ceil(nrows / step) - 1):
                for c in range(0, step_ncols - 1):
                    v1 = index
                    v2 = v1 + step_ncols
                    v3 = v2 + 1
                    v4 = v1 + 1
                    faces.append((v1, v2, v3, v4))
                    index += 1
                index += 1
        me = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, me)
        ob.location = (reprojection['to'].x - dx, reprojection['to'].y - dy, 0)
        scn = bpy.context.scene
        scn.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        me.from_pydata(vertices, [], faces)
        me.update()
        f.close()
        if prefs.adjust3Dview:
            bb = getBBOX.fromObj(ob)
            adjust3Dview(context, bb)
        return {'FINISHED'}

def register():
    if False:
        i = 10
        return i + 15
    try:
        bpy.utils.register_class(IMPORTGIS_OT_ascii_grid)
    except ValueError as e:
        log.warning('{} is already registered, now unregister and retry... '.format(IMPORTGIS_OT_ascii_grid))
        unregister()
        bpy.utils.register_class(IMPORTGIS_OT_ascii_grid)

def unregister():
    if False:
        for i in range(10):
            print('nop')
    bpy.utils.unregister_class(IMPORTGIS_OT_ascii_grid)