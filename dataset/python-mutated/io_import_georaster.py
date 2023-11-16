import bpy
import bmesh
import os
import math
from mathutils import Vector
import numpy as np
import logging
log = logging.getLogger(__name__)
from ..geoscene import GeoScene, georefManagerLayout
from ..prefs import PredefCRS
from ..core.georaster import GeoRaster
from .utils import bpyGeoRaster, exportAsMesh
from .utils import placeObj, adjust3Dview, showTextures, addTexture, getBBOX
from .utils import rasterExtentToMesh, geoRastUVmap, setDisplacer
from ..core import HAS_GDAL
if HAS_GDAL:
    from osgeo import gdal
from ..core import XY as xy
from ..core.errors import OverlapError
from ..core.proj import Reproj
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator
(PKG, SUBPKG) = __package__.split('.', maxsplit=1)

class IMPORTGIS_OT_georaster(Operator, ImportHelper):
    """Import georeferenced raster (need world file)"""
    bl_idname = 'importgis.georaster'
    bl_description = 'Import raster georeferenced with world file'
    bl_label = 'Import georaster'
    bl_options = {'UNDO'}

    def listObjects(self, context):
        if False:
            for i in range(10):
                print('nop')
        objs = []
        for (index, object) in enumerate(bpy.context.scene.objects):
            if object.type == 'MESH':
                objs.append((str(index), object.name, 'Object named ' + object.name))
        return objs
    filter_glob: StringProperty(default='*.tif;*.jpg;*.jpeg;*.png;*.bmp', options={'HIDDEN'})

    def listPredefCRS(self, context):
        if False:
            print('Hello World!')
        return PredefCRS.getEnumItems()
    rastCRS: EnumProperty(name='Raster CRS', description='Choose a Coordinate Reference System', items=listPredefCRS)
    reprojection: BoolProperty(name='Specifiy raster CRS', description="Specifiy raster CRS if it's different from scene CRS", default=False)
    importMode: EnumProperty(name='Mode', description='Select import mode', items=[('PLANE', 'Basemap on new plane', 'Place raster texture on new plane mesh'), ('BKG', 'Basemap as background', 'Place raster as background image'), ('MESH', 'Basemap on mesh', 'UV map raster on an existing mesh'), ('DEM', 'DEM as displacement texture', 'Use DEM raster as height texture to wrap a base mesh'), ('DEM_RAW', 'DEM raw data build [slow]', 'Import a DEM as pixels points cloud with building faces. Do not use with huge dataset.')])
    objectsLst: EnumProperty(attr='obj_list', name='Objects', description='Choose object to edit', items=listObjects)

    def listSubdivisionModes(self, context):
        if False:
            return 10
        items = [('subsurf', 'Subsurf', 'Add a subsurf modifier'), ('none', 'None', 'No subdivision')]
        if not self.demOnMesh:
            items.append(('mesh', 'Mesh', 'Create vertices at each pixels'))
        return items
    subdivision: EnumProperty(name='Subdivision', description='How to subdivise the plane (dispacer needs vertex to work with)', items=listSubdivisionModes)
    demOnMesh: BoolProperty(name='Apply on existing mesh', description='Use DEM as displacer for an existing mesh', default=False)
    clip: BoolProperty(name='Clip to working extent', description='Use the reference bounding box to clip the DEM', default=False)
    demInterpolation: BoolProperty(name='Smooth relief', description='Use texture interpolation to smooth the resulting terrain', default=True)
    fillNodata: BoolProperty(name='Fill nodata values', description='Interpolate existing nodata values to get an usuable displacement texture', default=False)
    step: IntProperty(name='Step', default=1, description='Pixel step', min=1)
    buildFaces: BoolProperty(name='Build faces', default=True, description='Build quad faces connecting pixel point cloud')

    def draw(self, context):
        if False:
            return 10
        layout = self.layout
        layout.prop(self, 'importMode')
        scn = bpy.context.scene
        geoscn = GeoScene(scn)
        if self.importMode == 'PLANE':
            pass
        if self.importMode == 'BKG':
            pass
        if self.importMode == 'MESH':
            if geoscn.isGeoref and len(self.objectsLst) > 0:
                layout.prop(self, 'objectsLst')
            else:
                layout.label(text="There isn't georef mesh to UVmap on")
        if self.importMode == 'DEM':
            layout.prop(self, 'demOnMesh')
            if self.demOnMesh:
                if geoscn.isGeoref and len(self.objectsLst) > 0:
                    layout.prop(self, 'objectsLst')
                    layout.prop(self, 'clip')
                else:
                    layout.label(text="There isn't georef mesh to apply on")
            layout.prop(self, 'subdivision')
            layout.prop(self, 'demInterpolation')
            if self.subdivision == 'mesh':
                layout.prop(self, 'step')
            layout.prop(self, 'fillNodata')
        if self.importMode == 'DEM_RAW':
            layout.prop(self, 'buildFaces')
            layout.prop(self, 'step')
            layout.prop(self, 'clip')
            if self.clip:
                if geoscn.isGeoref and len(self.objectsLst) > 0:
                    layout.prop(self, 'objectsLst')
                else:
                    layout.label(text="There isn't georef mesh to refer")
        if geoscn.isPartiallyGeoref:
            layout.prop(self, 'reprojection')
            if self.reprojection:
                self.crsInputLayout(context)
            georefManagerLayout(self, context)
        else:
            self.crsInputLayout(context)

    def crsInputLayout(self, context):
        if False:
            while True:
                i = 10
        layout = self.layout
        row = layout.row(align=True)
        split = row.split(factor=0.35, align=True)
        split.label(text='CRS:')
        split.prop(self, 'rastCRS', text='')
        row.operator('bgis.add_predef_crs', text='', icon='ADD')

    @classmethod
    def poll(cls, context):
        if False:
            return 10
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
        scale = geoscn.scale
        if geoscn.isGeoref:
            (dx, dy) = geoscn.getOriginPrj()
            if self.reprojection:
                rastCRS = self.rastCRS
            else:
                rastCRS = geoscn.crs
        else:
            rastCRS = self.rastCRS
            try:
                geoscn.crs = rastCRS
            except Exception as e:
                log.error('Cannot set scene crs', exc_info=True)
                self.report({'ERROR'}, 'Cannot set scene crs, check logs for more infos')
                return {'CANCELLED'}
        if geoscn.crs != rastCRS:
            rprj = True
            rprjToRaster = Reproj(geoscn.crs, rastCRS)
            rprjToScene = Reproj(rastCRS, geoscn.crs)
        else:
            rprj = False
            rprjToRaster = None
            rprjToScene = None
        filePath = self.filepath
        name = os.path.basename(filePath)[:-4]
        if self.importMode == 'PLANE':
            try:
                rast = bpyGeoRaster(filePath)
            except IOError as e:
                log.error('Unable to open raster', exc_info=True)
                self.report({'ERROR'}, 'Unable to open raster, check logs for more infos')
                return {'CANCELLED'}
            if not geoscn.isGeoref:
                (dx, dy) = (rast.center.x, rast.center.y)
                if rprj:
                    (dx, dy) = rprjToScene.pt(dx, dy)
                geoscn.setOriginPrj(dx, dy)
            mesh = rasterExtentToMesh(name, rast, dx, dy, reproj=rprjToScene)
            obj = placeObj(mesh, name)
            uvTxtLayer = mesh.uv_layers.new(name='rastUVmap')
            geoRastUVmap(obj, uvTxtLayer, rast, dx, dy, reproj=rprjToRaster)
            mat = bpy.data.materials.new('rastMat')
            obj.data.materials.append(mat)
            addTexture(mat, rast.bpyImg, uvTxtLayer, name='rastText')
        if self.importMode == 'BKG':
            if rprj:
                self.report({'ERROR'}, 'Raster reprojection is not possible in background mode')
                return {'CANCELLED'}
            try:
                rast = bpyGeoRaster(filePath)
            except IOError as e:
                log.error('Unable to open raster', exc_info=True)
                self.report({'ERROR'}, 'Unable to open raster, check logs for more infos')
                return {'CANCELLED'}
            if rast.rotation.xy != [0, 0]:
                self.report({'ERROR'}, 'Cannot apply a rotation in background image mode')
                return {'CANCELLED'}
            if abs(round(rast.pxSize.x, 3)) != abs(round(rast.pxSize.y, 3)):
                self.report({'ERROR'}, 'Background image needs equal pixel size in map units in both x ans y axis')
                return {'CANCELLED'}
            trueSizeX = rast.geoSize.x
            trueSizeY = rast.geoSize.y
            ratio = rast.size.x / rast.size.y
            if geoscn.isGeoref:
                (offx, offy) = (rast.center.x - dx, rast.center.y - dy)
            else:
                (dx, dy) = (rast.center.x, rast.center.y)
                geoscn.setOriginPrj(dx, dy)
                (offx, offy) = (0, 0)
            bkg = bpy.data.objects.new(self.name, None)
            bkg.empty_display_type = 'IMAGE'
            bkg.empty_image_depth = 'BACK'
            bkg.data = rast.bpyImg
            scn.collection.objects.link(bkg)
            bkg.empty_display_size = 1
            bkg.scale = (trueSizeX, trueSizeY * ratio, 1)
            bkg.location = (offx, offy, 0)
            bpy.context.view_layer.objects.active = bkg
            bkg.select_set(True)
            if prefs.adjust3Dview:
                adjust3Dview(context, rast.bbox)
        if self.importMode == 'MESH':
            if not geoscn.isGeoref or len(self.objectsLst) == 0:
                self.report({'ERROR'}, "There isn't georef mesh to apply on")
                return {'CANCELLED'}
            obj = scn.objects[int(self.objectsLst)]
            obj.select_set(True)
            context.view_layer.objects.active = obj
            subBox = getBBOX.fromObj(obj).toGeo(geoscn)
            if rprj:
                subBox = rprjToRaster.bbox(subBox)
            try:
                rast = bpyGeoRaster(filePath, subBoxGeo=subBox)
            except IOError as e:
                log.error('Unable to open raster', exc_info=True)
                self.report({'ERROR'}, 'Unable to open raster, check logs for more infos')
                return {'CANCELLED'}
            except OverlapError:
                self.report({'ERROR'}, 'Non overlap data')
                return {'CANCELLED'}
            mesh = obj.data
            uvTxtLayer = mesh.uv_layers.new(name='rastUVmap')
            uvTxtLayer.active = True
            geoRastUVmap(obj, uvTxtLayer, rast, dx, dy, reproj=rprjToRaster)
            mat = bpy.data.materials.new('rastMat')
            obj.data.materials.append(mat)
            addTexture(mat, rast.bpyImg, uvTxtLayer, name='rastText')
        if self.importMode == 'DEM':
            if self.demOnMesh:
                if not geoscn.isGeoref or len(self.objectsLst) == 0:
                    self.report({'ERROR'}, "There isn't georef mesh to apply on")
                    return {'CANCELLED'}
                obj = scn.objects[int(self.objectsLst)]
                mesh = obj.data
                obj.select_set(True)
                context.view_layer.objects.active = obj
                subBox = getBBOX.fromObj(obj).toGeo(geoscn)
                if rprj:
                    subBox = rprjToRaster.bbox(subBox)
            else:
                subBox = None
            try:
                grid = bpyGeoRaster(filePath, subBoxGeo=subBox, clip=self.clip, fillNodata=self.fillNodata, useGDAL=HAS_GDAL, raw=True)
            except IOError as e:
                log.error('Unable to open raster', exc_info=True)
                self.report({'ERROR'}, 'Unable to open raster, check logs for more infos')
                return {'CANCELLED'}
            except OverlapError:
                self.report({'ERROR'}, 'Non overlap data')
                return {'CANCELLED'}
            if not self.demOnMesh:
                if not geoscn.isGeoref:
                    (dx, dy) = (grid.center.x, grid.center.y)
                    if rprj:
                        (dx, dy) = rprjToScene.pt(dx, dy)
                    geoscn.setOriginPrj(dx, dy)
                if self.subdivision == 'mesh':
                    mesh = exportAsMesh(grid, dx, dy, self.step, reproj=rprjToScene, flat=True)
                else:
                    mesh = rasterExtentToMesh(name, grid, dx, dy, pxLoc='CENTER', reproj=rprjToScene)
                obj = placeObj(mesh, name)
                subBox = getBBOX.fromObj(obj).toGeo(geoscn)
            previousUVmapIdx = mesh.uv_layers.active_index
            uvTxtLayer = mesh.uv_layers.new(name='demUVmap')
            geoRastUVmap(obj, uvTxtLayer, grid, dx, dy, reproj=rprjToRaster)
            if previousUVmapIdx != -1:
                mesh.uv_layers.active_index = previousUVmapIdx
            if self.subdivision == 'subsurf':
                if not 'SUBSURF' in [mod.type for mod in obj.modifiers]:
                    subsurf = obj.modifiers.new('DEM', type='SUBSURF')
                    subsurf.subdivision_type = 'SIMPLE'
                    subsurf.levels = 6
                    subsurf.render_levels = 6
            dsp = setDisplacer(obj, grid, uvTxtLayer, interpolation=self.demInterpolation)
        if self.importMode == 'DEM_RAW':
            subBox = None
            if self.clip:
                if not geoscn.isGeoref or len(self.objectsLst) == 0:
                    self.report({'ERROR'}, 'No working extent')
                    return {'CANCELLED'}
                obj = scn.objects[int(self.objectsLst)]
                subBox = getBBOX.fromObj(obj).toGeo(geoscn)
                if rprj:
                    subBox = rprjToRaster.bbox(subBox)
            try:
                grid = GeoRaster(filePath, subBoxGeo=subBox, useGDAL=HAS_GDAL)
            except IOError as e:
                log.error('Unable to open raster', exc_info=True)
                self.report({'ERROR'}, 'Unable to open raster, check logs for more infos')
                return {'CANCELLED'}
            except OverlapError:
                self.report({'ERROR'}, 'Non overlap data')
                return {'CANCELLED'}
            if not geoscn.isGeoref:
                (dx, dy) = (grid.center.x, grid.center.y)
                if rprj:
                    (dx, dy) = rprjToScene.pt(dx, dy)
                geoscn.setOriginPrj(dx, dy)
            mesh = exportAsMesh(grid, dx, dy, self.step, reproj=rprjToScene, subset=self.clip, flat=False, buildFaces=self.buildFaces)
            obj = placeObj(mesh, name)
        if self.importMode == 'PLANE' or (self.importMode == 'DEM' and (not self.demOnMesh)) or self.importMode == 'DEM_RAW':
            newObjCreated = True
        else:
            newObjCreated = False
        if newObjCreated and prefs.adjust3Dview:
            bb = getBBOX.fromObj(obj)
            adjust3Dview(context, bb)
        if prefs.forceTexturedSolid:
            showTextures(context)
        return {'FINISHED'}

def register():
    if False:
        while True:
            i = 10
    try:
        bpy.utils.register_class(IMPORTGIS_OT_georaster)
    except ValueError as e:
        log.warning('{} is already registered, now unregister and retry... '.format(IMPORTGIS_OT_georaster))
        unregister()
        bpy.utils.register_class(IMPORTGIS_OT_georaster)

def unregister():
    if False:
        while True:
            i = 10
    bpy.utils.unregister_class(IMPORTGIS_OT_georaster)