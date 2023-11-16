import logging
log = logging.getLogger(__name__)
import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from .prefs import PredefCRS
from .core.proj.reproj import reprojPt
from .core.proj.srs import SRS
from .operators.utils import mouseTo3d
PKG = __package__
"\nPolicy :\nThis module manages in priority the CRS coordinates of the scene's origin and\nupdates the corresponding longitude/latitude only if it can to do the math.\n\nA scene is considered correctly georeferenced when at least a valid CRS is defined\nand the coordinates of scene's origin in this CRS space is set. A geoscene will be\nbroken if the origin is set but not the CRS or if the origin is only set as longitude/latitude.\n\nChanging the CRS will raise an error if updating existing origin coordinate is not possible.\n\nBoth methods setOriginGeo() and setOriginPrj() try a projection task to maintain\ncoordinates synchronized. Failing reprojection does not abort the exec, but will\ntrigger deletion of unsynch coordinates. Synchronization can be disable for\nsetOriginPrj() method only.\n\nExcept setOriginGeo() method, dealing directly with longitude/latitude\nautomatically trigger a reprojection task which will raise an error if failing.\n\nSequences of methods :\nmoveOriginPrj() | updOriginPrj() > setOriginPrj() > [reprojPt()]\nmoveOriginGeo() > updOriginGeo() > reprojPt() > updOriginPrj() > setOriginPrj()\n\nStandalone properties (lon, lat, crsx et crsy) can be edited independently without any extra checks.\n"

class SK:
    """Alias to Scene Keys used to store georef infos"""
    LAT = 'latitude'
    LON = 'longitude'
    CRS = 'SRID'
    CRSX = 'crs x'
    CRSY = 'crs y'
    SCALE = 'scale'
    ZOOM = 'zoom'

class GeoScene:

    def __init__(self, scn=None):
        if False:
            for i in range(10):
                print('nop')
        if scn is None:
            self.scn = bpy.context.scene
        else:
            self.scn = scn
        self.SK = SK()

    @property
    def _rna_ui(self):
        if False:
            print('Hello World!')
        rna_ui = self.scn.get('_RNA_UI', None)
        if rna_ui is None:
            self.scn['_RNA_UI'] = {}
            rna_ui = self.scn['_RNA_UI']
        return rna_ui

    def view3dToProj(self, dx, dy):
        if False:
            return 10
        'Convert view3d coords to crs coords'
        if self.hasOriginPrj:
            x = self.crsx + dx * self.scale
            y = self.crsy + dy * self.scale
            return (x, y)
        else:
            raise Exception('Scene origin coordinate is unset')

    def projToView3d(self, dx, dy):
        if False:
            i = 10
            return i + 15
        'Convert view3d coords to crs coords'
        if self.hasOriginPrj:
            x = dx * self.scale - self.crsx
            y = dy * self.scale - self.crsy
            return (x, y)
        else:
            raise Exception('Scene origin coordinate is unset')

    @property
    def hasCRS(self):
        if False:
            print('Hello World!')
        return SK.CRS in self.scn

    @property
    def hasValidCRS(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.hasCRS:
            return False
        return SRS.validate(self.crs)

    @property
    def isGeoref(self):
        if False:
            print('Hello World!')
        "A scene is georef if at least a valid CRS is defined and\n\t\tthe coordinates of scene's origin in this CRS space is set"
        return self.hasValidCRS and self.hasOriginPrj

    @property
    def isFullyGeoref(self):
        if False:
            print('Hello World!')
        return self.hasValidCRS and self.hasOriginPrj and self.hasOriginGeo

    @property
    def isPartiallyGeoref(self):
        if False:
            i = 10
            return i + 15
        return self.hasCRS or self.hasOriginPrj or self.hasOriginGeo

    @property
    def isBroken(self):
        if False:
            i = 10
            return i + 15
        'partial georef infos make the geoscene unusuable and broken'
        return self.hasCRS and (not self.hasValidCRS) or (not self.hasCRS and (self.hasOriginPrj or self.hasOriginGeo)) or (self.hasCRS and self.hasOriginGeo and (not self.hasOriginPrj))

    @property
    def hasOriginGeo(self):
        if False:
            for i in range(10):
                print('nop')
        return SK.LAT in self.scn and SK.LON in self.scn

    @property
    def hasOriginPrj(self):
        if False:
            i = 10
            return i + 15
        return SK.CRSX in self.scn and SK.CRSY in self.scn

    def setOriginGeo(self, lon, lat):
        if False:
            i = 10
            return i + 15
        (self.lon, self.lat) = (lon, lat)
        try:
            (self.crsx, self.crsy) = reprojPt(4326, self.crs, lon, lat)
        except Exception as e:
            if self.hasOriginPrj:
                self.delOriginPrj()
                log.warning('Origin proj has been deleted because the property could not be updated', exc_info=True)

    def setOriginPrj(self, x, y, synch=True):
        if False:
            for i in range(10):
                print('nop')
        (self.crsx, self.crsy) = (x, y)
        if synch:
            try:
                (self.lon, self.lat) = reprojPt(self.crs, 4326, x, y)
            except Exception as e:
                if self.hasOriginGeo:
                    self.delOriginGeo()
                    log.warning('Origin geo has been deleted because the property could not be updated', exc_info=True)
        elif self.hasOriginGeo:
            self.delOriginGeo()
            log.warning('Origin geo has been deleted because coordinate synchronization is disable')

    def updOriginPrj(self, x, y, updObjLoc=True, synch=True):
        if False:
            return 10
        'Update/move scene origin passing absolute coordinates'
        if not self.hasOriginPrj:
            raise Exception('Cannot update an unset origin.')
        dx = x - self.crsx
        dy = y - self.crsy
        self.setOriginPrj(x, y, synch)
        if updObjLoc:
            self._moveObjLoc(dx, dy)

    def updOriginGeo(self, lon, lat, updObjLoc=True):
        if False:
            while True:
                i = 10
        if not self.isGeoref:
            raise Exception('Cannot update geo origin of an ungeoref scene.')
        (x, y) = reprojPt(4326, self.crs, lon, lat)
        self.updOriginPrj(x, y, updObjLoc)

    def moveOriginGeo(self, dx, dy, updObjLoc=True):
        if False:
            while True:
                i = 10
        if not self.hasOriginGeo:
            raise Exception('Cannot move an unset origin.')
        x = self.lon + dx
        y = self.lat + dy
        self.updOriginGeo(x, y, updObjLoc=updObjLoc)

    def moveOriginPrj(self, dx, dy, useScale=True, updObjLoc=True, synch=True):
        if False:
            return 10
        'Move scene origin passing relative deltas'
        if not self.hasOriginPrj:
            raise Exception('Cannot move an unset origin.')
        if useScale:
            self.setOriginPrj(self.crsx + dx * self.scale, self.crsy + dy * self.scale, synch)
        else:
            self.setOriginPrj(self.crsx + dx, self.crsy + dy, synch)
        if updObjLoc:
            self._moveObjLoc(dx, dy)

    def _moveObjLoc(self, dx, dy):
        if False:
            i = 10
            return i + 15
        topParents = [obj for obj in self.scn.objects if not obj.parent]
        for obj in topParents:
            obj.location.x -= dx
            obj.location.y -= dy

    def getOriginGeo(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.lon, self.lat)

    def getOriginPrj(self):
        if False:
            while True:
                i = 10
        return (self.crsx, self.crsy)

    def delOriginGeo(self):
        if False:
            for i in range(10):
                print('nop')
        del self.lat
        del self.lon

    def delOriginPrj(self):
        if False:
            while True:
                i = 10
        del self.crsx
        del self.crsy

    def delOrigin(self):
        if False:
            while True:
                i = 10
        self.delOriginGeo()
        self.delOriginPrj()

    @property
    def crs(self):
        if False:
            while True:
                i = 10
        return self.scn.get(SK.CRS, None)

    @crs.setter
    def crs(self, v):
        if False:
            while True:
                i = 10
        crs = SRS(v)
        if self.hasOriginGeo:
            if crs.isWGS84:
                (self.crsx, self.crsy) = (self.lon, self.lat)
            (self.crsx, self.crsy) = reprojPt(4326, str(crs), self.lon, self.lat)
        elif self.hasOriginPrj and self.hasCRS:
            if self.hasValidCRS:
                (self.crsx, self.crsy) = reprojPt(self.crs, str(crs), self.crsx, self.crsy)
            else:
                raise Exception('Scene origin coordinates cannot be updated because current CRS is invalid.')
        if SK.CRS not in self.scn:
            self._rna_ui[SK.CRS] = {'description': 'Map Coordinate Reference System', 'default': ''}
        self.scn[SK.CRS] = str(crs)

    @crs.deleter
    def crs(self):
        if False:
            for i in range(10):
                print('nop')
        if SK.CRS in self.scn:
            del self.scn[SK.CRS]

    @property
    def lat(self):
        if False:
            while True:
                i = 10
        return self.scn.get(SK.LAT, None)

    @lat.setter
    def lat(self, v):
        if False:
            return 10
        if SK.LAT not in self.scn:
            self._rna_ui[SK.LAT] = {'description': 'Scene origin latitude', 'default': 0.0, 'min': -90.0, 'max': 90.0}
        if -90 <= v <= 90:
            self.scn[SK.LAT] = v
        else:
            raise ValueError('Wrong latitude value ' + str(v))

    @lat.deleter
    def lat(self):
        if False:
            print('Hello World!')
        if SK.LAT in self.scn:
            del self.scn[SK.LAT]

    @property
    def lon(self):
        if False:
            print('Hello World!')
        return self.scn.get(SK.LON, None)

    @lon.setter
    def lon(self, v):
        if False:
            for i in range(10):
                print('nop')
        if SK.LON not in self.scn:
            self._rna_ui[SK.LON] = {'description': 'Scene origin longitude', 'default': 0.0, 'min': -180.0, 'max': 180.0}
        if -180 <= v <= 180:
            self.scn[SK.LON] = v
        else:
            raise ValueError('Wrong longitude value ' + str(v))

    @lon.deleter
    def lon(self):
        if False:
            i = 10
            return i + 15
        if SK.LON in self.scn:
            del self.scn[SK.LON]

    @property
    def crsx(self):
        if False:
            return 10
        return self.scn.get(SK.CRSX, None)

    @crsx.setter
    def crsx(self, v):
        if False:
            while True:
                i = 10
        if SK.CRSX not in self.scn:
            self._rna_ui[SK.CRSX] = {'description': 'Scene x origin in CRS space', 'default': 0.0}
        if isinstance(v, (int, float)):
            self.scn[SK.CRSX] = v
        else:
            raise ValueError('Wrong x origin value ' + str(v))

    @crsx.deleter
    def crsx(self):
        if False:
            return 10
        if SK.CRSX in self.scn:
            del self.scn[SK.CRSX]

    @property
    def crsy(self):
        if False:
            return 10
        return self.scn.get(SK.CRSY, None)

    @crsy.setter
    def crsy(self, v):
        if False:
            i = 10
            return i + 15
        if SK.CRSY not in self.scn:
            self._rna_ui[SK.CRSY] = {'description': 'Scene y origin in CRS space', 'default': 0.0}
        if isinstance(v, (int, float)):
            self.scn[SK.CRSY] = v
        else:
            raise ValueError('Wrong y origin value ' + str(v))

    @crsy.deleter
    def crsy(self):
        if False:
            print('Hello World!')
        if SK.CRSY in self.scn:
            del self.scn[SK.CRSY]

    @property
    def scale(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scn.get(SK.SCALE, 1)

    @scale.setter
    def scale(self, v):
        if False:
            return 10
        if SK.SCALE not in self.scn:
            self._rna_ui[SK.SCALE] = {'description': 'Map scale denominator', 'default': 1, 'min': 1}
        self.scn[SK.SCALE] = v

    @scale.deleter
    def scale(self):
        if False:
            while True:
                i = 10
        if SK.SCALE in self.scn:
            del self.scn[SK.SCALE]

    @property
    def zoom(self):
        if False:
            while True:
                i = 10
        return self.scn.get(SK.ZOOM, None)

    @zoom.setter
    def zoom(self, v):
        if False:
            print('Hello World!')
        if SK.ZOOM not in self.scn:
            self._rna_ui[SK.ZOOM] = {'description': 'Basemap zoom level', 'default': 1, 'min': 0, 'max': 25}
        self.scn[SK.ZOOM] = v

    @zoom.deleter
    def zoom(self):
        if False:
            return 10
        if SK.ZOOM in self.scn:
            del self.scn[SK.ZOOM]

    @property
    def hasScale(self):
        if False:
            print('Hello World!')
        return SK.SCALE in self.scn

    @property
    def hasZoom(self):
        if False:
            return 10
        return self.zoom is not None
from bpy_extras.view3d_utils import region_2d_to_location_3d, region_2d_to_vector_3d

class GEOSCENE_OT_coords_viewer(Operator):
    bl_idname = 'geoscene.coords_viewer'
    bl_description = ''
    bl_label = ''
    bl_options = {'INTERNAL', 'UNDO'}
    coords: FloatVectorProperty(subtype='XYZ')

    @classmethod
    def poll(cls, context):
        if False:
            print('Hello World!')
        return bpy.context.mode == 'OBJECT' and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        if False:
            for i in range(10):
                print('nop')
        self.geoscn = GeoScene(context.scene)
        if not self.geoscn.isGeoref or self.geoscn.isBroken:
            self.report({'ERROR'}, 'Scene is not correctly georeferencing')
            return {'CANCELLED'}
        context.window_manager.modal_handler_add(self)
        self.timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window.cursor_set('CROSSHAIR')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if False:
            i = 10
            return i + 15
        if event.type == 'MOUSEMOVE':
            loc = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
            (x, y) = self.geoscn.view3dToProj(loc.x, loc.y)
            context.area.header_text_set('x {:.3f}, y {:.3f}, z {:.3f}'.format(x, y, loc.z))
        if event.type == 'ESC' and event.value == 'PRESS':
            context.window.cursor_set('DEFAULT')
            context.area.header_text_set(None)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

class GEOSCENE_OT_set_crs(Operator):
    """
	use the enum of predefinites crs defined in addon prefs
	to select and switch scene crs definition
	"""
    bl_idname = 'geoscene.set_crs'
    bl_description = 'Switch scene crs'
    bl_label = 'Switch to'
    bl_options = {'INTERNAL', 'UNDO'}
    '\n\t#to avoid conflict, make a distinct predef crs enum\n\t#instead of reuse the one defined in addon pref\n\n\tdef listPredefCRS(self, context):\n\t\treturn PredefCRS.getEnumItems()\n\n\tcrsEnum = EnumProperty(\n\t\tname = "Predefinate CRS",\n\t\tdescription = "Choose predefinite Coordinate Reference System",\n\t\titems = listPredefCRS\n\t\t)\n\t'

    def draw(self, context):
        if False:
            return 10
        prefs = context.preferences.addons[PKG].preferences
        layout = self.layout
        row = layout.row(align=True)
        row.prop(prefs, 'predefCrs', text='')
        row.operator('bgis.add_predef_crs', text='', icon='ADD')

    def invoke(self, context, event):
        if False:
            return 10
        return context.window_manager.invoke_props_dialog(self, width=200)

    def execute(self, context):
        if False:
            while True:
                i = 10
        geoscn = GeoScene(context.scene)
        prefs = context.preferences.addons[PKG].preferences
        try:
            geoscn.crs = prefs.predefCrs
        except Exception as err:
            log.error('Cannot update crs', exc_info=True)
            self.report({'ERROR'}, 'Cannot update crs. Check logs form more info')
            return {'CANCELLED'}
        context.area.tag_redraw()
        return {'FINISHED'}

class GEOSCENE_OT_init_org(Operator):
    bl_idname = 'geoscene.init_org'
    bl_description = 'Init scene origin custom props at location 0,0'
    bl_label = 'Init origin'
    bl_options = {'INTERNAL', 'UNDO'}
    lonlat: BoolProperty(name='As lonlat', description='Set origin coordinate as longitude and latitude')
    x: FloatProperty()
    y: FloatProperty()

    def invoke(self, context, event):
        if False:
            return 10
        return context.window_manager.invoke_props_dialog(self, width=200)

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        geoscn = GeoScene(context.scene)
        if geoscn.hasOriginGeo or geoscn.hasOriginPrj:
            log.warning('Cannot init scene origin because it already exist')
            return {'CANCELLED'}
        elif self.lonlat:
            geoscn.setOriginGeo(self.x, self.y)
        else:
            geoscn.setOriginPrj(self.x, self.y)
        return {'FINISHED'}

class GEOSCENE_OT_edit_org_geo(Operator):
    bl_idname = 'geoscene.edit_org_geo'
    bl_description = 'Edit scene origin longitude/latitude'
    bl_label = 'Edit origin geo'
    bl_options = {'INTERNAL', 'UNDO'}
    lon: FloatProperty()
    lat: FloatProperty()

    def invoke(self, context, event):
        if False:
            for i in range(10):
                print('nop')
        geoscn = GeoScene(context.scene)
        if geoscn.isBroken:
            self.report({'ERROR'}, 'Scene georef is broken')
            return {'CANCELLED'}
        (self.lon, self.lat) = geoscn.getOriginGeo()
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if False:
            return 10
        geoscn = GeoScene(context.scene)
        if geoscn.hasOriginGeo:
            geoscn.updOriginGeo(self.lon, self.lat)
        else:
            geoscn.setOriginGeo(self.lon, self.lat)
        return {'FINISHED'}

class GEOSCENE_OT_edit_org_prj(Operator):
    bl_idname = 'geoscene.edit_org_prj'
    bl_description = 'Edit scene origin in projected system'
    bl_label = 'Edit origin proj'
    bl_options = {'INTERNAL', 'UNDO'}
    x: FloatProperty()
    y: FloatProperty()

    def invoke(self, context, event):
        if False:
            for i in range(10):
                print('nop')
        geoscn = GeoScene(context.scene)
        if geoscn.isBroken:
            self.report({'ERROR'}, 'Scene georef is broken')
            return {'CANCELLED'}
        (self.x, self.y) = geoscn.getOriginPrj()
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if False:
            print('Hello World!')
        geoscn = GeoScene(context.scene)
        if geoscn.hasOriginPrj:
            geoscn.updOriginPrj(self.x, self.y)
        else:
            geoscn.setOriginPrj(self.x, self.y)
        return {'FINISHED'}

class GEOSCENE_OT_link_org_geo(Operator):
    bl_idname = 'geoscene.link_org_geo'
    bl_description = 'Link scene origin lat long'
    bl_label = 'Link geo'
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        if False:
            for i in range(10):
                print('nop')
        geoscn = GeoScene(context.scene)
        if geoscn.hasOriginPrj and geoscn.hasCRS:
            try:
                (geoscn.lon, geoscn.lat) = reprojPt(geoscn.crs, 4326, geoscn.crsx, geoscn.crsy)
            except Exception as err:
                log.error('Cannot compute lat/lon coordinates', exc_info=True)
                self.report({'ERROR'}, 'Cannot compute lat/lon. Check logs for more infos.')
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, 'No enough infos')
            return {'CANCELLED'}
        return {'FINISHED'}

class GEOSCENE_OT_link_org_prj(Operator):
    bl_idname = 'geoscene.link_org_prj'
    bl_description = 'Link scene origin in crs space'
    bl_label = 'Link prj'
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        if False:
            return 10
        geoscn = GeoScene(context.scene)
        if geoscn.hasOriginGeo and geoscn.hasCRS:
            try:
                (geoscn.crsx, geoscn.crsy) = reprojPt(4326, geoscn.crs, geoscn.lon, geoscn.lat)
            except Exception as err:
                log.error('Cannot compute crs coordinates', exc_info=True)
                self.report({'ERROR'}, 'Cannot compute crs coordinates. Check logs for more infos.')
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, 'No enough infos')
            return {'CANCELLED'}
        return {'FINISHED'}

class GEOSCENE_OT_clear_org(Operator):
    bl_idname = 'geoscene.clear_org'
    bl_description = 'Clear scene origin coordinates'
    bl_label = 'Clear origin'
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        if False:
            while True:
                i = 10
        geoscn = GeoScene(context.scene)
        geoscn.delOrigin()
        return {'FINISHED'}

class GEOSCENE_OT_clear_georef(Operator):
    bl_idname = 'geoscene.clear_georef'
    bl_description = 'Clear all georef infos'
    bl_label = 'Clear georef'
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        geoscn = GeoScene(context.scene)
        geoscn.delOrigin()
        del geoscn.crs
        return {'FINISHED'}

def getLon(self):
    if False:
        print('Hello World!')
    geoscn = GeoScene()
    return geoscn.lon

def getLat(self):
    if False:
        return 10
    geoscn = GeoScene()
    return geoscn.lat

def setLon(self, lon):
    if False:
        while True:
            i = 10
    geoscn = GeoScene()
    prefs = bpy.context.preferences.addons[PKG].preferences
    if geoscn.hasOriginGeo:
        geoscn.updOriginGeo(lon, geoscn.lat, updObjLoc=prefs.lockObj)
    else:
        geoscn.setOriginGeo(lon, geoscn.lat)

def setLat(self, lat):
    if False:
        i = 10
        return i + 15
    geoscn = GeoScene()
    prefs = bpy.context.preferences.addons[PKG].preferences
    if geoscn.hasOriginGeo:
        geoscn.updOriginGeo(geoscn.lon, lat, updObjLoc=prefs.lockObj)
    else:
        geoscn.setOriginGeo(geoscn.lon, lat)

def getCrsx(self):
    if False:
        return 10
    geoscn = GeoScene()
    return geoscn.crsx

def getCrsy(self):
    if False:
        return 10
    geoscn = GeoScene()
    return geoscn.crsy

def setCrsx(self, x):
    if False:
        i = 10
        return i + 15
    geoscn = GeoScene()
    prefs = bpy.context.preferences.addons[PKG].preferences
    if geoscn.hasOriginPrj:
        geoscn.updOriginPrj(x, geoscn.crsy, updObjLoc=prefs.lockObj)
    else:
        geoscn.setOriginPrj(x, geoscn.crsy)

def setCrsy(self, y):
    if False:
        return 10
    geoscn = GeoScene()
    prefs = bpy.context.preferences.addons[PKG].preferences
    if geoscn.hasOriginPrj:
        geoscn.updOriginPrj(geoscn.crsx, y, updObjLoc=prefs.lockObj)
    else:
        geoscn.setOriginPrj(geoscn.crsx, y)

class GEOSCENE_PT_georef(Panel):
    bl_category = 'View'
    bl_label = 'Geoscene'
    bl_space_type = 'VIEW_3D'
    bl_context = 'objectmode'
    bl_region_type = 'UI'

    def draw(self, context):
        if False:
            print('Hello World!')
        layout = self.layout
        scn = context.scene
        geoscn = GeoScene(scn)
        georefManagerLayout(self, context)
        layout.operator('geoscene.coords_viewer', icon='WORLD', text='Geo-coordinates')

class GLOBAL_PROPS(PropertyGroup):
    displayOriginGeo: BoolProperty(name='Geo', description='Display longitude and latitude of scene origin')
    displayOriginPrj: BoolProperty(name='Proj', description='Display coordinates of scene origin in CRS space')
    lon: FloatProperty(get=getLon, set=setLon)
    lat: FloatProperty(get=getLat, set=setLat)
    crsx: FloatProperty(get=getCrsx, set=setCrsx)
    crsy: FloatProperty(get=getCrsy, set=setCrsy)

def georefManagerLayout(self, context):
    if False:
        print('Hello World!')
    'Use this method to extend a panel with georef managment tools'
    layout = self.layout
    scn = context.scene
    wm = bpy.context.window_manager
    geoscn = GeoScene(scn)
    prefs = context.preferences.addons[PKG].preferences
    if geoscn.isBroken:
        layout.alert = True
    row = layout.row(align=True)
    row.label(text='Scene georeferencing :')
    if geoscn.hasCRS:
        row.operator('geoscene.clear_georef', text='', icon='CANCEL')
    row = layout.row(align=True)
    split = row.split(factor=0.25)
    if geoscn.hasCRS:
        split.label(icon='PROP_ON', text='CRS:')
    elif not geoscn.hasCRS and (geoscn.hasOriginGeo or geoscn.hasOriginPrj):
        split.label(icon='ERROR', text='CRS:')
    else:
        split.label(icon='PROP_OFF', text='CRS:')
    if geoscn.hasCRS:
        crs = scn[SK.CRS]
        name = PredefCRS.getName(crs)
        if name is not None:
            split.label(text=name)
        else:
            split.label(text=crs)
    else:
        split.label(text='Not set')
    row.operator('geoscene.set_crs', text='', icon='PREFERENCES')
    row = layout.row(align=True)
    split = row.split(factor=0.25, align=True)
    if not geoscn.hasOriginGeo and (not geoscn.hasOriginPrj):
        split.label(icon='PROP_OFF', text='Origin:')
    elif not geoscn.hasOriginGeo and geoscn.hasOriginPrj:
        split.label(icon='PROP_CON', text='Origin:')
    elif geoscn.hasOriginGeo and geoscn.hasOriginPrj:
        split.label(icon='PROP_ON', text='Origin:')
    elif geoscn.hasOriginGeo and (not geoscn.hasOriginPrj):
        split.label(icon='ERROR', text='Origin:')
    col = split.column(align=True)
    if not geoscn.hasOriginGeo:
        col.enabled = False
    col.prop(wm.geoscnProps, 'displayOriginGeo', toggle=True)
    col = split.column(align=True)
    if not geoscn.hasOriginPrj:
        col.enabled = False
    col.prop(wm.geoscnProps, 'displayOriginPrj', toggle=True)
    if geoscn.hasOriginGeo or geoscn.hasOriginPrj:
        if geoscn.hasCRS and (not geoscn.hasOriginPrj):
            row.operator('geoscene.link_org_prj', text='', icon='CONSTRAINT')
        if geoscn.hasCRS and (not geoscn.hasOriginGeo):
            row.operator('geoscene.link_org_geo', text='', icon='CONSTRAINT')
        row.operator('geoscene.clear_org', text='', icon='REMOVE')
    if not geoscn.hasOriginGeo and (not geoscn.hasOriginPrj):
        row.operator('geoscene.init_org', text='', icon='ADD')
    if geoscn.hasOriginGeo and wm.geoscnProps.displayOriginGeo:
        row = layout.row()
        row.prop(wm.geoscnProps, 'lon', text='Lon')
        row.prop(wm.geoscnProps, 'lat', text='Lat')
        '\n\t\trow.enabled = False\n\t\trow.prop(scn, \'["\'+SK.LON+\'"]\', text=\'Lon\')\n\t\trow.prop(scn, \'["\'+SK.LAT+\'"]\', text=\'Lat\')\n\t\t'
    if geoscn.hasOriginPrj and wm.geoscnProps.displayOriginPrj:
        row = layout.row()
        row.prop(wm.geoscnProps, 'crsx', text='X')
        row.prop(wm.geoscnProps, 'crsy', text='Y')
        '\n\t\trow.enabled = False\n\t\trow.prop(scn, \'["\'+SK.CRSX+\'"]\', text=\'X\')\n\t\trow.prop(scn, \'["\'+SK.CRSY+\'"]\', text=\'Y\')\n\t\t'
    if geoscn.hasScale:
        row = layout.row()
        row.label(text='Map scale:')
        col = row.column()
        col.enabled = False
        col.prop(scn, '["' + SK.SCALE + '"]', text='')
classes = [GEOSCENE_OT_coords_viewer, GEOSCENE_OT_set_crs, GEOSCENE_OT_init_org, GEOSCENE_OT_edit_org_geo, GEOSCENE_OT_edit_org_prj, GEOSCENE_OT_link_org_geo, GEOSCENE_OT_link_org_prj, GEOSCENE_OT_clear_org, GEOSCENE_OT_clear_georef, GEOSCENE_PT_georef, GLOBAL_PROPS]

def register():
    if False:
        print('Hello World!')
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            log.warning('{} is already registered, now unregister and retry... '.format(cls))
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.WindowManager.geoscnProps = PointerProperty(type=GLOBAL_PROPS)

def unregister():
    if False:
        print('Hello World!')
    del bpy.types.WindowManager.geoscnProps
    for cls in classes:
        bpy.utils.unregister_class(cls)