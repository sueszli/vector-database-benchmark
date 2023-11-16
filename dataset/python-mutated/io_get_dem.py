import os
import time
import logging
log = logging.getLogger(__name__)
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty
from ..geoscene import GeoScene
from .utils import adjust3Dview, getBBOX, isTopView
from ..core.proj import SRS, reprojBbox
from ..core import settings
USER_AGENT = settings.user_agent
(PKG, SUBPKG) = __package__.split('.', maxsplit=1)
TIMEOUT = 120

class IMPORTGIS_OT_dem_query(Operator):
    """Import elevation data from a web service"""
    bl_idname = 'importgis.dem_query'
    bl_description = 'Query for elevation data from a web service'
    bl_label = 'Get elevation (SRTM)'
    bl_options = {'UNDO'}

    def invoke(self, context, event):
        if False:
            print('Hello World!')
        geoscn = GeoScene(context.scene)
        if not geoscn.isGeoref:
            self.report({'ERROR'}, 'Scene is not georef')
            return {'CANCELLED'}
        if geoscn.isBroken:
            self.report({'ERROR'}, 'Scene georef is broken, please fix it beforehand')
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        if False:
            return 10
        prefs = context.preferences.addons[PKG].preferences
        layout = self.layout
        row = layout.row(align=True)
        row.prop(prefs, 'demServer', text='Server')
        if 'opentopography' in prefs.demServer:
            row = layout.row(align=True)
            row.prop(prefs, 'opentopography_api_key', text='Api Key')

    @classmethod
    def poll(cls, context):
        if False:
            while True:
                i = 10
        return context.mode == 'OBJECT'

    def execute(self, context):
        if False:
            return 10
        prefs = bpy.context.preferences.addons[PKG].preferences
        scn = context.scene
        geoscn = GeoScene(scn)
        crs = SRS(geoscn.crs)
        objs = bpy.context.selected_objects
        aObj = context.active_object
        if len(objs) == 1 and aObj.type == 'MESH':
            onMesh = True
            bbox = getBBOX.fromObj(aObj).toGeo(geoscn)
        elif isTopView(context):
            onMesh = False
            bbox = getBBOX.fromTopView(context).toGeo(geoscn)
        else:
            self.report({'ERROR'}, 'Please define the query extent in orthographic top view or by selecting a reference object')
            return {'CANCELLED'}
        if bbox.dimensions.x > 1000000 or bbox.dimensions.y > 1000000:
            self.report({'ERROR'}, 'Too large extent')
            return {'CANCELLED'}
        bbox = reprojBbox(geoscn.crs, 4326, bbox)
        if 'SRTM' in prefs.demServer:
            if bbox.ymin > 60:
                self.report({'ERROR'}, 'SRTM is not available beyond 60 degrees north')
                return {'CANCELLED'}
            if bbox.ymax < -56:
                self.report({'ERROR'}, 'SRTM is not available below 56 degrees south')
                return {'CANCELLED'}
        if 'opentopography' in prefs.demServer:
            if not prefs.opentopography_api_key:
                self.report({'ERROR'}, 'Please register to opentopography.org and request for an API key')
                return {'CANCELLED'}
        w = context.window
        w.cursor_set('WAIT')
        e = 0.002
        (xmin, xmax) = (bbox.xmin - e, bbox.xmax + e)
        (ymin, ymax) = (bbox.ymin - e, bbox.ymax + e)
        url = prefs.demServer.format(W=xmin, E=xmax, S=ymin, N=ymax, API_KEY=prefs.opentopography_api_key)
        log.debug(url)
        if bpy.data.is_saved:
            filePath = os.path.join(os.path.dirname(bpy.data.filepath), 'srtm.tif')
        else:
            filePath = os.path.join(bpy.app.tempdir, 'srtm.tif')
        rq = Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urlopen(rq, timeout=TIMEOUT) as response, open(filePath, 'wb') as outFile:
                data = response.read()
                outFile.write(data)
        except (URLError, HTTPError) as err:
            log.error('Http request fails url:{}, code:{}, error:{}'.format(url, getattr(err, 'code', None), err.reason))
            self.report({'ERROR'}, 'Cannot reach OpenTopography web service, check logs for more infos')
            return {'CANCELLED'}
        except TimeoutError:
            log.error('Http request does not respond. url:{}, code:{}, error:{}'.format(url, getattr(err, 'code', None), err.reason))
            info = 'Cannot reach SRTM web service provider, server can be down or overloaded. Please retry later'
            log.info(info)
            self.report({'ERROR'}, info)
            return {'CANCELLED'}
        if not onMesh:
            bpy.ops.importgis.georaster('EXEC_DEFAULT', filepath=filePath, reprojection=True, rastCRS='EPSG:4326', importMode='DEM', subdivision='subsurf', demInterpolation=True)
        else:
            bpy.ops.importgis.georaster('EXEC_DEFAULT', filepath=filePath, reprojection=True, rastCRS='EPSG:4326', importMode='DEM', subdivision='subsurf', demInterpolation=True, demOnMesh=True, objectsLst=[str(i) for (i, obj) in enumerate(scn.collection.all_objects) if obj.name == bpy.context.active_object.name][0], clip=False, fillNodata=False)
        bbox = getBBOX.fromScn(scn)
        adjust3Dview(context, bbox, zoomToSelect=False)
        return {'FINISHED'}

def register():
    if False:
        i = 10
        return i + 15
    try:
        bpy.utils.register_class(IMPORTGIS_OT_dem_query)
    except ValueError as e:
        log.warning('{} is already registered, now unregister and retry... '.format(IMPORTGIS_OT_srtm_query))
        unregister()
        bpy.utils.register_class(IMPORTGIS_OT_dem_query)

def unregister():
    if False:
        i = 10
        return i + 15
    bpy.utils.unregister_class(IMPORTGIS_OT_dem_query)