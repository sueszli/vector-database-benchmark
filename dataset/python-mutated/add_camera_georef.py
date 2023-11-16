import logging
log = logging.getLogger(__name__)
import bpy
from mathutils import Vector
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from .utils import getBBOX
from ..geoscene import GeoScene

class CAMERA_OT_add_georender_cam(bpy.types.Operator):
    """
	Add a new georef camera or update an existing one
	A georef camera is a top view orthographic camera that can be used to render a map
	The camera is setting to encompass the selected object, the output spatial resolution (meters/pixel) can be set by the user
	A worldfile is writen in BLender text editor, it can be used to georef the output render
	"""
    bl_idname = 'camera.georender'
    bl_label = 'Georef cam'
    bl_description = 'Create or update a camera to render a georeferencing map'
    bl_options = {'REGISTER', 'UNDO'}
    name: StringProperty(name='Camera name', default='Georef cam', description='')
    target_res: FloatProperty(name='Pixel size', default=5, description='Pixel size in map units/pixel', min=1e-05)
    zLocOffset: FloatProperty(name='Z loc. off.', default=50, description='Camera z location offet, defined as percentage of z dimension of the target mesh', min=0)
    redo = 0
    bbox = None

    def check(self, context):
        if False:
            for i in range(10):
                print('nop')
        return True

    def draw(self, context):
        if False:
            i = 10
            return i + 15
        layout = self.layout
        layout.prop(self, 'name')
        layout.prop(self, 'target_res')
        layout.prop(self, 'zLocOffset')

    @classmethod
    def poll(cls, context):
        if False:
            i = 10
            return i + 15
        return context.mode == 'OBJECT'

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        self.redo += 1
        scn = context.scene
        geoscn = GeoScene(scn)
        if not geoscn.isGeoref:
            self.report({'ERROR'}, "Scene isn't georef")
            return {'CANCELLED'}
        objs = bpy.context.selected_objects
        if (not objs or len(objs) > 2) or (len(objs) == 1 and (not objs[0].type == 'MESH')) or (len(objs) == 2 and (not set((objs[0].type, objs[1].type)) == set(('MESH', 'CAMERA')))):
            self.report({'ERROR'}, 'Pre-selection is incorrect')
            return {'CANCELLED'}
        if len(objs) == 2:
            newCam = False
        else:
            newCam = True
        (dx, dy) = geoscn.getOriginPrj()
        for obj in objs:
            if obj.type == 'MESH':
                georefObj = obj
            elif obj.type == 'CAMERA':
                camObj = obj
                cam = camObj.data
        if self.bbox is None:
            bbox = getBBOX.fromObj(georefObj, applyTransform=True)
            self.bbox = bbox
        else:
            bbox = self.bbox
        (locx, locy, locz) = bbox.center
        (dimx, dimy, dimz) = bbox.dimensions
        if dimz == 0:
            dimz = 1
        if newCam:
            cam = bpy.data.cameras.new(name=self.name)
            cam['mapRes'] = self.target_res
            camObj = bpy.data.objects.new(name=self.name, object_data=cam)
            scn.collection.objects.link(camObj)
            scn.camera = camObj
        elif self.redo == 1:
            scn.camera = camObj
            try:
                self.target_res = cam['mapRes']
            except KeyError:
                self.report({'ERROR'}, 'This camera has not map resolution property')
                return {'CANCELLED'}
        else:
            try:
                cam['mapRes'] = self.target_res
            except KeyError:
                self.report({'ERROR'}, 'This camera has not map resolution property')
                return {'CANCELLED'}
        cam.type = 'ORTHO'
        cam.ortho_scale = max((dimx, dimy))
        offset = dimz * self.zLocOffset / 100
        camLocZ = bbox['zmin'] + dimz + offset
        camObj.location = (locx, locy, camLocZ)
        cam.clip_start = 0
        cam.clip_end = dimz + offset * 2
        cam.show_limits = True
        if not newCam:
            if self.redo == 1:
                self.name = camObj.name
            else:
                camObj.name = self.name
                camObj.data.name = self.name
        bpy.ops.object.select_all(action='DESELECT')
        camObj.select_set(True)
        context.view_layer.objects.active = camObj
        scn.camera = camObj
        scn.render.resolution_x = int(dimx / self.target_res)
        scn.render.resolution_y = int(dimy / self.target_res)
        scn.render.resolution_percentage = 100
        res = self.target_res
        rot = 0
        x = bbox['xmin'] + dx
        y = bbox['ymax'] + dy
        wf_data = '\n'.join(map(str, [res, rot, rot, -res, x + res / 2, y - res / 2]))
        wf_name = camObj.name + '.wld'
        if wf_name in bpy.data.texts:
            wfText = bpy.data.texts[wf_name]
            wfText.clear()
        else:
            wfText = bpy.data.texts.new(name=wf_name)
        wfText.write(wf_data)
        for wfText in bpy.data.texts:
            (name, ext) = (wfText.name[:-4], wfText.name[-4:])
            if ext == '.wld' and name not in bpy.data.objects:
                bpy.data.texts.remove(wfText)
        return {'FINISHED'}

def register():
    if False:
        for i in range(10):
            print('nop')
    try:
        bpy.utils.register_class(CAMERA_OT_add_georender_cam)
    except ValueError as e:
        log.warning('{} is already registered, now unregister and retry... '.format(CAMERA_OT_add_georender_cam))
        unregister()
        bpy.utils.register_class(CAMERA_OT_add_georender_cam)

def unregister():
    if False:
        return 10
    bpy.utils.unregister_class(CAMERA_OT_add_georender_cam)