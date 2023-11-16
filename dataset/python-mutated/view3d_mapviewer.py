import math
import os
import threading
import logging
log = logging.getLogger(__name__)
import bpy
from mathutils import Vector
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty
import addon_utils
import gpu
from gpu_extras.batch import batch_for_shader
from ..core import HAS_GDAL, HAS_PIL, HAS_IMGIO
from ..core.proj import reprojPt, reprojBbox, dd2meters, meters2dd
from ..core.basemaps import GRIDS, SOURCES, MapService
from ..core import settings
USER_AGENT = settings.user_agent
from ..geoscene import GeoScene, SK, georefManagerLayout
from ..prefs import PredefCRS
from .utils import getBBOX, mouseTo3d
from .utils import placeObj, adjust3Dview, showTextures, rasterExtentToMesh, geoRastUVmap, addTexture
from .lib.osm.nominatim import nominatimQuery
(PKG, SUBPKG) = __package__.split('.', maxsplit=1)

class BaseMap(GeoScene):
    """Handle a map as background image in Blender"""

    def __init__(self, context, srckey, laykey, grdkey=None):
        if False:
            return 10
        self.context = context
        self.scn = context.scene
        GeoScene.__init__(self, self.scn)
        self.area = context.area
        self.area3d = [r for r in self.area.regions if r.type == 'WINDOW'][0]
        self.view3d = self.area.spaces.active
        self.reg3d = self.view3d.region_3d
        prefs = context.preferences.addons[PKG].preferences
        cacheFolder = prefs.cacheFolder
        self.synchOrj = prefs.synchOrj
        MapService.RESAMP_ALG = prefs.resamplAlg
        self.srv = MapService(srckey, cacheFolder)
        self.name = srckey + '_' + laykey + '_' + grdkey
        if grdkey is None:
            grdkey = self.srv.srcGridKey
        if grdkey == self.srv.srcGridKey:
            self.tm = self.srv.srcTms
        else:
            self.srv.setDstGrid(grdkey)
            self.tm = self.srv.dstTms
        if not self.hasCRS:
            self.crs = self.tm.CRS
        if not self.hasOriginPrj:
            self.setOriginPrj(0, 0, self.synchOrj)
        if not self.hasScale:
            self.scale = 1
        if not self.hasZoom:
            self.zoom = 0
        self.lockedZoom = None
        if bpy.data.is_saved:
            folder = os.path.dirname(bpy.data.filepath) + os.sep
        else:
            folder = bpy.app.tempdir
        self.imgPath = folder + self.name + '.tif'
        self.layer = self.srv.layers[laykey]
        self.srckey = srckey
        self.laykey = laykey
        self.grdkey = grdkey
        self.thread = None
        self.img = None
        self.bkg = None
        self.viewDstZ = None

    def get(self):
        if False:
            return 10
        'Launch run() function in a new thread'
        self.stop()
        self.srv.start()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stop actual thread'
        if self.srv.running:
            self.srv.stop()
            self.thread.join()

    def run(self):
        if False:
            print('Hello World!')
        'thread method'
        self.mosaic = self.request()
        if self.srv.running and self.mosaic is not None:
            self.mosaic.save(self.imgPath)
        if self.srv.running:
            self.place()
        self.srv.stop()

    def moveOrigin(self, dx, dy, useScale=True, updObjLoc=True):
        if False:
            while True:
                i = 10
        'Move scene origin and update props'
        self.moveOriginPrj(dx, dy, useScale, updObjLoc, self.synchOrj)

    def request(self):
        if False:
            while True:
                i = 10
        'Request map service to build a mosaic of required tiles to cover view3d area'
        (w, h) = (self.area.width, self.area.height)
        z = self.lockedZoom if self.lockedZoom is not None else self.zoom
        res = self.tm.getRes(z)
        if self.crs == 'EPSG:4326':
            res = meters2dd(res)
        (dx, dy, dz) = self.reg3d.view_location
        ox = self.crsx + dx * self.scale
        oy = self.crsy + dy * self.scale
        xmin = ox - w / 2 * res * self.scale
        ymax = oy + h / 2 * res * self.scale
        xmax = ox + w / 2 * res * self.scale
        ymin = oy - h / 2 * res * self.scale
        bbox = (xmin, ymin, xmax, ymax)
        if self.crs != self.tm.CRS:
            bbox = reprojBbox(self.crs, self.tm.CRS, bbox)
        '\n\t\t#Method 2\n\t\tbbox = getBBOX.fromTopView(self.context) #ERROR context is None ????\n\t\tbbox = bbox.toGeo(geoscn=self)\n\t\tif self.crs != self.tm.CRS:\n\t\t\tbbox = reprojBbox(self.crs, self.tm.CRS, bbox)\n\t\t'
        log.debug('Bounding box request : {}'.format(bbox))
        if self.srv.srcGridKey == self.grdkey:
            toDstGrid = False
        else:
            toDstGrid = True
        mosaic = self.srv.getImage(self.laykey, bbox, self.zoom, toDstGrid=toDstGrid, outCRS=self.crs)
        return mosaic

    def place(self):
        if False:
            for i in range(10):
                print('nop')
        'Set map as background image'
        try:
            self.img = [img for img in bpy.data.images if img.filepath == self.imgPath and len(img.packed_files) == 0][0]
        except IndexError:
            self.img = bpy.data.images.load(self.imgPath)
        empties = [obj for obj in self.scn.objects if obj.type == 'EMPTY']
        bkgs = [obj for obj in empties if obj.empty_display_type == 'IMAGE']
        for bkg in bkgs:
            bkg.hide_viewport = True
        try:
            self.bkg = [bkg for bkg in bkgs if bkg.data.filepath == self.imgPath and len(bkg.data.packed_files) == 0][0]
        except IndexError:
            self.bkg = bpy.data.objects.new(self.name, None)
            self.bkg.empty_display_type = 'IMAGE'
            self.bkg.empty_image_depth = 'BACK'
            self.bkg.data = self.img
            self.scn.collection.objects.link(self.bkg)
        else:
            self.bkg.hide_viewport = False
        (img_ox, img_oy) = self.mosaic.center
        (img_w, img_h) = self.mosaic.size
        res = self.mosaic.pxSize.x
        sizex = img_w * res / self.scale
        sizey = img_h * res / self.scale
        size = max([sizex, sizey])
        self.bkg.empty_display_size = 1
        self.bkg.scale = (size, size, 1)
        dx = (self.crsx - img_ox) / self.scale
        dy = (self.crsy - img_oy) / self.scale
        self.bkg.location = (-dx, -dy, 0)
        dst = max([self.area.width, self.area.height])
        z = self.lockedZoom if self.lockedZoom is not None else self.zoom
        res = self.tm.getRes(z)
        dst = dst * res / self.scale
        view3D_aperture = 36
        view3D_zoom = 2
        fov = 2 * math.atan(view3D_aperture / (self.view3d.lens * 2))
        fov = math.atan(math.tan(fov / 2) * view3D_zoom) * 2
        zdst = dst / 2 / math.tan(fov / 2)
        zdst = math.floor(zdst)
        self.reg3d.view_distance = zdst
        self.viewDstZ = zdst
        self.bkg.data.reload()

def drawInfosText(self, context):
    if False:
        print('Hello World!')
    scn = context.scene
    area = context.area
    area3d = [reg for reg in area.regions if reg.type == 'WINDOW'][0]
    view3d = area.spaces.active
    reg3d = view3d.region_3d
    geoscn = GeoScene(scn)
    zoom = geoscn.zoom
    scale = geoscn.scale
    txt = 'Map view : '
    txt += 'Zoom ' + str(zoom)
    if self.map.lockedZoom is not None:
        txt += ' (Locked)'
    txt += ' - Scale 1:' + str(int(scale))
    "\n\t# view3d distance\n\tdst = reg3d.view_distance\n\tif dst > 1000:\n\t\tdst /= 1000\n\t\tunit = 'km'\n\telse:\n\t\tunit = 'm'\n\ttxt += ' 3D View distance ' + str(int(dst)) + ' ' + unit\n\t"
    txt += ' ' + str((int(self.posx), int(self.posy)))
    txt += ' ' + self.progress
    context.area.header_text_set(txt)

def drawZoomBox(self, context):
    if False:
        while True:
            i = 10
    if self.zoomBoxMode and (not self.zoomBoxDrag):
        (px, py) = (self.zb_xmax, self.zb_ymax)
        p1 = (0, py, 0)
        p2 = (context.area.width, py, 0)
        p3 = (px, 0, 0)
        p4 = (px, context.area.height, 0)
        coords = [p1, p2, p3, p4]
        shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {'pos': coords})
        shader.bind()
        shader.uniform_float('color', (0, 0, 0, 1))
        batch.draw(shader)
    elif self.zoomBoxMode and self.zoomBoxDrag:
        p1 = (self.zb_xmin, self.zb_ymin, 0)
        p2 = (self.zb_xmin, self.zb_ymax, 0)
        p3 = (self.zb_xmax, self.zb_ymax, 0)
        p4 = (self.zb_xmax, self.zb_ymin, 0)
        coords = [p1, p2, p2, p3, p3, p4, p4, p1]
        shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {'pos': coords})
        shader.bind()
        shader.uniform_float('color', (0, 0, 0, 1))
        batch.draw(shader)

class VIEW3D_OT_map_start(Operator):
    bl_idname = 'view3d.map_start'
    bl_description = 'Toggle 2d map navigation'
    bl_label = 'Basemap'
    bl_options = {'REGISTER'}

    def check(self, context):
        if False:
            for i in range(10):
                print('nop')
        return True

    def listSources(self, context):
        if False:
            i = 10
            return i + 15
        srcItems = []
        for (srckey, src) in SOURCES.items():
            srcItems.append((srckey, src['name'], src['description']))
        return srcItems

    def listGrids(self, context):
        if False:
            return 10
        grdItems = []
        src = SOURCES[self.src]
        for (gridkey, grd) in GRIDS.items():
            if gridkey == src['grid']:
                grdItems.insert(0, (gridkey, grd['name'] + ' (source)', grd['description']))
            else:
                grdItems.append((gridkey, grd['name'], grd['description']))
        return grdItems

    def listLayers(self, context):
        if False:
            while True:
                i = 10
        layItems = []
        src = SOURCES[self.src]
        for (laykey, lay) in src['layers'].items():
            layItems.append((laykey, lay['name'], lay['description']))
        return layItems
    src: EnumProperty(name='Map', description='Choose map service source', items=listSources)
    grd: EnumProperty(name='Grid', description='Choose cache tiles matrix', items=listGrids)
    lay: EnumProperty(name='Layer', description='Choose layer', items=listLayers)
    dialog: StringProperty(default='MAP')
    query: StringProperty(name='Go to')
    zoom: IntProperty(name='Zoom level', min=0, max=25)
    recenter: BoolProperty(name='Center to existing objects')

    def draw(self, context):
        if False:
            print('Hello World!')
        addonPrefs = context.preferences.addons[PKG].preferences
        scn = context.scene
        layout = self.layout
        if self.dialog == 'SEARCH':
            layout.prop(self, 'query')
            layout.prop(self, 'zoom', slider=True)
        elif self.dialog == 'OPTIONS':
            layout.prop(addonPrefs, 'zoomToMouse')
            layout.prop(addonPrefs, 'lockObj')
            layout.prop(addonPrefs, 'lockOrigin')
            layout.prop(addonPrefs, 'synchOrj')
        elif self.dialog == 'MAP':
            layout.prop(self, 'src', text='Source')
            layout.prop(self, 'lay', text='Layer')
            col = layout.column()
            if not HAS_GDAL:
                col.enabled = False
                col.label(text='(No raster reprojection support)')
            col.prop(self, 'grd', text='Tile matrix set')
            grdCRS = GRIDS[self.grd]['CRS']
            row = layout.row()
            desc = PredefCRS.getName(grdCRS)
            if desc is not None:
                row.label(text='CRS: ' + desc)
            else:
                row.label(text='CRS: ' + grdCRS)
            row = layout.row()
            row.prop(self, 'recenter')
            geoscn = GeoScene(scn)
            if geoscn.isPartiallyGeoref:
                georefManagerLayout(self, context)

    def invoke(self, context, event):
        if False:
            return 10
        if not HAS_PIL and (not HAS_GDAL) and (not HAS_IMGIO):
            self.report({'ERROR'}, 'No imaging library available. ImageIO module was not correctly installed.')
            return {'CANCELLED'}
        if not context.area.type == 'VIEW_3D':
            self.report({'WARNING'}, 'View3D not found, cannot run operator')
            return {'CANCELLED'}
        geoscn = GeoScene(context.scene)
        if geoscn.hasZoom:
            self.zoom = geoscn.zoom
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if False:
            print('Hello World!')
        scn = context.scene
        geoscn = GeoScene(scn)
        prefs = context.preferences.addons[PKG].preferences
        folder = prefs.cacheFolder
        if folder == '' or not os.path.exists(folder):
            self.report({'ERROR'}, "Please define a valid cache folder path in addon's preferences")
            return {'CANCELLED'}
        if not os.access(folder, os.X_OK | os.W_OK):
            self.report({'ERROR'}, 'The selected cache folder has no write access')
            return {'CANCELLED'}
        if self.dialog == 'MAP':
            grdCRS = GRIDS[self.grd]['CRS']
            if geoscn.isBroken:
                self.report({'ERROR'}, 'Scene georef is broken, please fix it beforehand')
                return {'CANCELLED'}
            if geoscn.hasCRS and geoscn.crs != grdCRS and (not HAS_GDAL):
                self.report({'ERROR'}, 'Please install gdal to enable raster reprojection support')
                return {'CANCELLED'}
        if self.dialog == 'SEARCH':
            r = bpy.ops.view3d.map_search('EXEC_DEFAULT', query=self.query)
            if r == {'CANCELLED'}:
                self.report({'INFO'}, 'No location found')
            else:
                geoscn.zoom = self.zoom
        self.dialog = 'MAP'
        bpy.ops.view3d.map_viewer('INVOKE_DEFAULT', srckey=self.src, laykey=self.lay, grdkey=self.grd, recenter=self.recenter)
        return {'FINISHED'}

class VIEW3D_OT_map_viewer(Operator):
    bl_idname = 'view3d.map_viewer'
    bl_description = 'Toggle 2d map navigation'
    bl_label = 'Map viewer'
    bl_options = {'INTERNAL'}
    srckey: StringProperty()
    grdkey: StringProperty()
    laykey: StringProperty()
    recenter: BoolProperty()

    @classmethod
    def poll(cls, context):
        if False:
            print('Hello World!')
        return context.area.type == 'VIEW_3D'

    def __del__(self):
        if False:
            return 10
        if getattr(self, 'restart', False):
            bpy.ops.view3d.map_start('INVOKE_DEFAULT', src=self.srckey, lay=self.laykey, grd=self.grdkey, dialog=self.dialog)

    def invoke(self, context, event):
        if False:
            for i in range(10):
                print('nop')
        self.restart = False
        self.dialog = 'MAP'
        self.moveFactor = 0.1
        self.prefs = context.preferences.addons[PKG].preferences
        self.updObjLoc = self.prefs.lockObj
        args = (self, context)
        self._drawTextHandler = bpy.types.SpaceView3D.draw_handler_add(drawInfosText, args, 'WINDOW', 'POST_PIXEL')
        self._drawZoomBoxHandler = bpy.types.SpaceView3D.draw_handler_add(drawZoomBox, args, 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        self.timer = context.window_manager.event_timer_add(0.04, window=context.window)
        view3d = context.area.spaces.active
        bpy.ops.view3d.view_axis(type='TOP')
        view3d.region_3d.view_perspective = 'ORTHO'
        context.scene.cursor.location = (0, 0, 0)
        if not self.prefs.lockOrigin:
            view3d.region_3d.view_location = (0, 0, 0)
        self.inMove = False
        (self.posx, self.posy) = (0, 0)
        self.progress = ''
        self.zoomBoxMode = False
        self.zoomBoxDrag = False
        (self.zb_xmin, self.zb_xmax) = (0, 0)
        (self.zb_ymin, self.zb_ymax) = (0, 0)
        self.map = BaseMap(context, self.srckey, self.laykey, self.grdkey)
        if self.recenter and len(context.scene.objects) > 0:
            scnBbox = getBBOX.fromScn(context.scene).to2D()
            (w, h) = scnBbox.dimensions
            px_diag = math.sqrt(context.area.width ** 2 + context.area.height ** 2)
            dst_diag = math.sqrt(w ** 2 + h ** 2)
            targetRes = dst_diag / px_diag
            z = self.map.tm.getNearestZoom(targetRes, rule='lower')
            resFactor = self.map.tm.getFromToResFac(self.map.zoom, z)
            context.region_data.view_distance *= resFactor
            (x, y) = scnBbox.center
            if self.prefs.lockOrigin:
                context.region_data.view_location = (x, y, 0)
            else:
                self.map.moveOrigin(x, y)
            self.map.zoom = z
        self.map.get()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if False:
            print('Hello World!')
        context.area.tag_redraw()
        scn = bpy.context.scene
        if event.type == 'TIMER':
            self.progress = self.map.srv.report
            return {'PASS_THROUGH'}
        if event.type in ['WHEELUPMOUSE', 'NUMPAD_PLUS']:
            if event.value == 'PRESS':
                if event.alt:
                    self.map.scale *= 10
                    self.map.place()
                    for obj in scn.objects:
                        obj.location /= 10
                        obj.scale /= 10
                elif event.ctrl:
                    dst = context.region_data.view_distance
                    context.region_data.view_distance -= dst * self.moveFactor
                    if self.prefs.zoomToMouse:
                        mouseLoc = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
                        viewLoc = context.region_data.view_location
                        deltaVect = (mouseLoc - viewLoc) * self.moveFactor
                        viewLoc += deltaVect
                elif self.map.zoom < self.map.layer.zmax and self.map.zoom < self.map.tm.nbLevels - 1:
                    self.map.zoom += 1
                    if self.map.lockedZoom is None:
                        resFactor = self.map.tm.getNextResFac(self.map.zoom)
                        if not self.prefs.zoomToMouse:
                            context.region_data.view_distance *= resFactor
                        else:
                            dst = context.region_data.view_distance
                            dst2 = dst * resFactor
                            context.region_data.view_distance = dst2
                            mouseLoc = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
                            viewLoc = context.region_data.view_location
                            moveFactor = (dst - dst2) / dst
                            deltaVect = (mouseLoc - viewLoc) * moveFactor
                            if self.prefs.lockOrigin:
                                viewLoc += deltaVect
                            else:
                                (dx, dy, dz) = deltaVect
                                if not self.prefs.lockObj and self.map.bkg is not None:
                                    self.map.bkg.location -= deltaVect
                                self.map.moveOrigin(dx, dy, updObjLoc=self.updObjLoc)
                    self.map.get()
        if event.type in ['WHEELDOWNMOUSE', 'NUMPAD_MINUS']:
            if event.value == 'PRESS':
                if event.alt:
                    s = self.map.scale / 10
                    if s < 1:
                        s = 1
                    self.map.scale = s
                    self.map.place()
                    for obj in scn.objects:
                        obj.location *= 10
                        obj.scale *= 10
                elif event.ctrl:
                    dst = context.region_data.view_distance
                    context.region_data.view_distance += dst * self.moveFactor
                    if self.prefs.zoomToMouse:
                        mouseLoc = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
                        viewLoc = context.region_data.view_location
                        deltaVect = (mouseLoc - viewLoc) * self.moveFactor
                        viewLoc -= deltaVect
                elif self.map.zoom > self.map.layer.zmin and self.map.zoom > 0:
                    self.map.zoom -= 1
                    if self.map.lockedZoom is None:
                        resFactor = self.map.tm.getPrevResFac(self.map.zoom)
                        if not self.prefs.zoomToMouse:
                            context.region_data.view_distance *= resFactor
                        else:
                            dst = context.region_data.view_distance
                            dst2 = dst * resFactor
                            context.region_data.view_distance = dst2
                            mouseLoc = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
                            viewLoc = context.region_data.view_location
                            moveFactor = (dst - dst2) / dst
                            deltaVect = (mouseLoc - viewLoc) * moveFactor
                            if self.prefs.lockOrigin:
                                viewLoc += deltaVect
                            else:
                                (dx, dy, dz) = deltaVect
                                if not self.prefs.lockObj and self.map.bkg is not None:
                                    self.map.bkg.location -= deltaVect
                                self.map.moveOrigin(dx, dy, updObjLoc=self.updObjLoc)
                    self.map.get()
        if event.type == 'MOUSEMOVE':
            loc = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
            (self.posx, self.posy) = self.map.view3dToProj(loc.x, loc.y)
            if self.zoomBoxMode:
                (self.zb_xmax, self.zb_ymax) = (event.mouse_region_x, event.mouse_region_y)
            if self.inMove:
                loc1 = mouseTo3d(context, self.x1, self.y1)
                loc2 = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
                dlt = loc1 - loc2
                if event.ctrl or self.prefs.lockOrigin:
                    context.region_data.view_location = self.viewLoc1 + dlt
                else:
                    if self.map.bkg is not None:
                        self.map.bkg.location[0] = self.offx1 - dlt.x
                        self.map.bkg.location[1] = self.offy1 - dlt.y
                    if self.updObjLoc:
                        topParents = [obj for obj in scn.objects if not obj.parent]
                        for (i, obj) in enumerate(topParents):
                            if obj == self.map.bkg:
                                continue
                            loc1 = self.objsLoc1[i]
                            obj.location.x = loc1.x - dlt.x
                            obj.location.y = loc1.y - dlt.y
        if event.type in {'LEFTMOUSE', 'MIDDLEMOUSE'}:
            if event.value == 'PRESS' and (not self.zoomBoxMode):
                (self.x1, self.y1) = (event.mouse_region_x, event.mouse_region_y)
                self.viewLoc1 = context.region_data.view_location.copy()
                if not event.ctrl:
                    self.map.stop()
                    if not self.prefs.lockOrigin:
                        if self.map.bkg is not None:
                            self.offx1 = self.map.bkg.location[0]
                            self.offy1 = self.map.bkg.location[1]
                        self.objsLoc1 = [obj.location.copy() for obj in scn.objects if not obj.parent]
                self.inMove = True
            if event.value == 'RELEASE' and (not self.zoomBoxMode):
                self.inMove = False
                if not event.ctrl:
                    if not self.prefs.lockOrigin:
                        loc1 = mouseTo3d(context, self.x1, self.y1)
                        loc2 = mouseTo3d(context, event.mouse_region_x, event.mouse_region_y)
                        dlt = loc1 - loc2
                        self.map.moveOrigin(dlt.x, dlt.y, updObjLoc=False)
                    self.map.get()
            if event.value == 'PRESS' and self.zoomBoxMode:
                self.zoomBoxDrag = True
                (self.zb_xmin, self.zb_ymin) = (event.mouse_region_x, event.mouse_region_y)
            if event.value == 'RELEASE' and self.zoomBoxMode:
                xmax = max(event.mouse_region_x, self.zb_xmin)
                ymax = max(event.mouse_region_y, self.zb_ymin)
                xmin = min(event.mouse_region_x, self.zb_xmin)
                ymin = min(event.mouse_region_y, self.zb_ymin)
                self.zoomBoxDrag = False
                self.zoomBoxMode = False
                context.window.cursor_set('DEFAULT')
                w = xmax - xmin
                h = ymax - ymin
                cx = xmin + w / 2
                cy = ymin + h / 2
                loc = mouseTo3d(context, cx, cy)
                px_diag = math.sqrt(context.area.width ** 2 + context.area.height ** 2)
                mapRes = self.map.tm.getRes(self.map.zoom)
                dst_diag = math.sqrt((w * mapRes) ** 2 + (h * mapRes) ** 2)
                targetRes = dst_diag / px_diag
                z = self.map.tm.getNearestZoom(targetRes, rule='lower')
                resFactor = self.map.tm.getFromToResFac(self.map.zoom, z)
                context.region_data.view_distance *= resFactor
                if self.prefs.lockOrigin:
                    context.region_data.view_location = loc
                else:
                    self.map.moveOrigin(loc.x, loc.y, updObjLoc=self.updObjLoc)
                self.map.zoom = z
                self.map.get()
        if event.type in ['LEFT_CTRL', 'RIGHT_CTRL']:
            if event.value == 'PRESS':
                self._viewDstZ = context.region_data.view_distance
                self._viewLoc = context.region_data.view_location.copy()
            if event.value == 'RELEASE':
                context.region_data.view_distance = self._viewDstZ
                context.region_data.view_location = self._viewLoc
        if event.value == 'PRESS' and event.type in ['NUMPAD_2', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_8']:
            delta = self.map.bkg.scale.x * self.moveFactor
            if event.type == 'NUMPAD_4':
                if event.ctrl or self.prefs.lockOrigin:
                    context.region_data.view_location += Vector((-delta, 0, 0))
                else:
                    self.map.moveOrigin(-delta, 0, updObjLoc=self.updObjLoc)
            if event.type == 'NUMPAD_6':
                if event.ctrl or self.prefs.lockOrigin:
                    context.region_data.view_location += Vector((delta, 0, 0))
                else:
                    self.map.moveOrigin(delta, 0, updObjLoc=self.updObjLoc)
            if event.type == 'NUMPAD_2':
                if event.ctrl or self.prefs.lockOrigin:
                    context.region_data.view_location += Vector((0, -delta, 0))
                else:
                    self.map.moveOrigin(0, -delta, updObjLoc=self.updObjLoc)
            if event.type == 'NUMPAD_8':
                if event.ctrl or self.prefs.lockOrigin:
                    context.region_data.view_location += Vector((0, delta, 0))
                else:
                    self.map.moveOrigin(0, delta, updObjLoc=self.updObjLoc)
            if not event.ctrl:
                self.map.get()
        if event.type == 'SPACE':
            self.map.stop()
            bpy.types.SpaceView3D.draw_handler_remove(self._drawTextHandler, 'WINDOW')
            bpy.types.SpaceView3D.draw_handler_remove(self._drawZoomBoxHandler, 'WINDOW')
            context.area.header_text_set(None)
            self.restart = True
            return {'FINISHED'}
        if event.type == 'G':
            self.map.stop()
            bpy.types.SpaceView3D.draw_handler_remove(self._drawTextHandler, 'WINDOW')
            bpy.types.SpaceView3D.draw_handler_remove(self._drawZoomBoxHandler, 'WINDOW')
            context.area.header_text_set(None)
            self.restart = True
            self.dialog = 'SEARCH'
            return {'FINISHED'}
        if event.type == 'O':
            self.map.stop()
            bpy.types.SpaceView3D.draw_handler_remove(self._drawTextHandler, 'WINDOW')
            bpy.types.SpaceView3D.draw_handler_remove(self._drawZoomBoxHandler, 'WINDOW')
            context.area.header_text_set(None)
            self.restart = True
            self.dialog = 'OPTIONS'
            return {'FINISHED'}
        if event.type == 'L' and event.value == 'PRESS':
            if self.map.lockedZoom is None:
                self.map.lockedZoom = self.map.zoom
            else:
                self.map.lockedZoom = None
                self.map.get()
        if event.type == 'B' and event.value == 'PRESS':
            self.map.stop()
            self.zoomBoxMode = True
            (self.zb_xmax, self.zb_ymax) = (event.mouse_region_x, event.mouse_region_y)
            context.window.cursor_set('CROSSHAIR')
        if event.type == 'E' and event.value == 'PRESS':
            if not self.map.srv.running and self.map.mosaic is not None:
                self.map.stop()
                self.map.bkg.hide_viewport = True
                bpy.types.SpaceView3D.draw_handler_remove(self._drawTextHandler, 'WINDOW')
                bpy.types.SpaceView3D.draw_handler_remove(self._drawZoomBoxHandler, 'WINDOW')
                context.area.header_text_set(None)
                bpyImg = bpy.data.images.load(self.map.imgPath)
                name = 'EXPORT_' + self.map.srckey + '_' + self.map.laykey + '_' + self.map.grdkey
                bpyImg.name = name
                bpyImg.pack()
                rast = self.map.mosaic
                setattr(rast, 'bpyImg', bpyImg)
                (dx, dy) = self.map.getOriginPrj()
                mesh = rasterExtentToMesh(name, rast, dx, dy, pxLoc='CORNER')
                obj = placeObj(mesh, name)
                uvTxtLayer = mesh.uv_layers.new(name='rastUVmap')
                geoRastUVmap(obj, uvTxtLayer, rast, dx, dy)
                mat = bpy.data.materials.new('rastMat')
                obj.data.materials.append(mat)
                addTexture(mat, bpyImg, uvTxtLayer)
                if self.prefs.adjust3Dview:
                    adjust3Dview(context, getBBOX.fromObj(obj))
                if self.prefs.forceTexturedSolid:
                    showTextures(context)
                return {'FINISHED'}
        if event.type == 'ESC' and event.value == 'PRESS':
            if self.zoomBoxMode:
                self.zoomBoxDrag = False
                self.zoomBoxMode = False
                context.window.cursor_set('DEFAULT')
            else:
                self.map.stop()
                bpy.types.SpaceView3D.draw_handler_remove(self._drawTextHandler, 'WINDOW')
                bpy.types.SpaceView3D.draw_handler_remove(self._drawZoomBoxHandler, 'WINDOW')
                context.area.header_text_set(None)
                return {'CANCELLED'}
        return {'RUNNING_MODAL'}

class VIEW3D_OT_map_search(bpy.types.Operator):
    bl_idname = 'view3d.map_search'
    bl_description = 'Search for a place and move scene origin to it'
    bl_label = 'Map search'
    bl_options = {'INTERNAL'}
    query: StringProperty(name='Go to')

    def invoke(self, context, event):
        if False:
            print('Hello World!')
        geoscn = GeoScene(context.scene)
        if geoscn.isBroken:
            self.report({'ERROR'}, 'Scene georef is broken')
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if False:
            for i in range(10):
                print('nop')
        geoscn = GeoScene(context.scene)
        prefs = context.preferences.addons[PKG].preferences
        try:
            results = nominatimQuery(self.query, referer='bgis', user_agent=USER_AGENT)
        except Exception as e:
            log.error('Failed Nominatim query', exc_info=True)
            return {'CANCELLED'}
        if len(results) == 0:
            return {'CANCELLED'}
        else:
            log.debug('Nominatim search results : {}'.format([r['display_name'] for r in results]))
            result = results[0]
            (lat, lon) = (float(result['lat']), float(result['lon']))
            if geoscn.isGeoref:
                geoscn.updOriginGeo(lon, lat, updObjLoc=prefs.lockObj)
            else:
                geoscn.setOriginGeo(lon, lat)
        return {'FINISHED'}
classes = [VIEW3D_OT_map_start, VIEW3D_OT_map_viewer, VIEW3D_OT_map_search]

def register():
    if False:
        for i in range(10):
            print('nop')
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            log.warning('{} is already registered, now unregister and retry... '.format(cls))
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)

def unregister():
    if False:
        i = 10
        return i + 15
    for cls in classes:
        bpy.utils.unregister_class(cls)