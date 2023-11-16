import os
from math import pi
import logging
log = logging.getLogger(__name__)
import bpy
from bpy.props import StringProperty, CollectionProperty, EnumProperty
from bpy.types import Panel, Operator, OperatorFileListElement
from ..geoscene import GeoScene
from ..core.proj import reprojPt
from ..core.georaster import getImgFormat
from ..core.lib import Tyf

def newEmpty(scene, name, location):
    if False:
        i = 10
        return i + 15
    'Create a new empty'
    target = bpy.data.objects.new(name, None)
    target.empty_display_size = 40
    target.empty_display_type = 'PLAIN_AXES'
    target.location = location
    scene.collection.objects.link(target)
    return target

def newCamera(scene, name, location, focalLength):
    if False:
        i = 10
        return i + 15
    'Create a new camera'
    cam = bpy.data.cameras.new(name)
    cam.sensor_width = 35
    cam.lens = focalLength
    cam.display_size = 40
    cam_obj = bpy.data.objects.new(name, cam)
    cam_obj.location = location
    cam_obj.rotation_euler[0] = pi / 2
    cam_obj.rotation_euler[2] = pi
    scene.collection.objects.link(cam_obj)
    return (cam, cam_obj)

def newTargetCamera(scene, name, location, focalLength):
    if False:
        i = 10
        return i + 15
    'Create a new camera.target'
    (cam, cam_obj) = newCamera(scene, name, location, focalLength)
    (x, y, z) = location[:]
    target = newEmpty(scene, name + '.target', (x, y - 50, z))
    constraint = cam_obj.constraints.new(type='TRACK_TO')
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    constraint.target = target
    return (cam, cam_obj)

class CAMERA_OT_geophotos_add(Operator):
    bl_idname = 'camera.geophotos'
    bl_description = 'Create cameras from geotagged photos'
    bl_label = 'Exif cam'
    bl_options = {'REGISTER'}
    files: CollectionProperty(name='File Path', type=OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')
    filter_glob: StringProperty(default='*.jpg;*.jpeg;*.tif;*.tiff', options={'HIDDEN'})
    filename_ext = ''
    exifMode: EnumProperty(attr='exif_mode', name='Action', description='Choose an action', items=[('TARGET_CAMERA', 'Target Camera', 'Create a camera with target helper'), ('CAMERA', 'Camera', 'Create a camera'), ('EMPTY', 'Empty', 'Create an empty helper'), ('CURSOR', 'Cursor', 'Move cursor')], default='TARGET_CAMERA')

    def invoke(self, context, event):
        if False:
            i = 10
            return i + 15
        scn = context.scene
        geoscn = GeoScene(scn)
        if not geoscn.isGeoref:
            self.report({'ERROR'}, 'The scene must be georeferenced.')
            return {'CANCELLED'}
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if False:
            while True:
                i = 10
        scn = context.scene
        geoscn = GeoScene(scn)
        directory = self.directory
        for file_elem in self.files:
            filepath = os.path.join(directory, file_elem.name)
            if not os.path.isfile(filepath):
                self.report({'ERROR'}, 'Invalid file')
                return {'CANCELLED'}
            imgFormat = getImgFormat(filepath)
            if imgFormat not in ['JPEG', 'TIFF']:
                self.report({'ERROR'}, 'Invalid format ' + str(imgFormat))
                return {'CANCELLED'}
            try:
                exif = Tyf.open(filepath)
            except Exception as e:
                log.error('Unable to open file', exc_info=True)
                self.report({'ERROR'}, 'Unable to open file. Checks logs for more infos.')
                return {'CANCELLED'}
            try:
                lat = exif['GPSLatitude'] * exif['GPSLatitudeRef']
                lon = exif['GPSLongitude'] * exif['GPSLongitudeRef']
            except KeyError:
                self.report({'ERROR'}, "Can't find GPS longitude or latitude.")
                return {'CANCELLED'}
            try:
                alt = exif['GPSAltitude']
            except KeyError:
                alt = 0
            try:
                (x, y) = reprojPt(4326, geoscn.crs, lon, lat)
            except Exception as e:
                log.error('Reprojection fails', exc_info=True)
                self.report({'ERROR'}, 'Reprojection error. Check logs for more infos.')
                return {'CANCELLED'}
            try:
                focalLength = exif['FocalLengthIn35mmFilm']
            except KeyError:
                focalLength = 35
            location = (x - geoscn.crsx, y - geoscn.crsy, alt)
            name = bpy.path.display_name_from_filepath(filepath)
            if self.exifMode == 'TARGET_CAMERA':
                (cam, cam_obj) = newTargetCamera(scn, name, location, focalLength)
            elif self.exifMode == 'CAMERA':
                (cam, cam_obj) = newCamera(scn, name, location, focalLength)
            elif self.exifMode == 'EMPTY':
                newEmpty(scn, name, location)
            else:
                scn.cursor.location = location
            if self.exifMode in ['TARGET_CAMERA', 'CAMERA']:
                cam['background'] = filepath
                '\n                try:\n                    cam[\'imageWidth\']  = exif["PixelXDimension"] #for jpg, in tif tag is named imageWidth...\n                    cam[\'imageHeight\'] = exif["PixelYDimension"]\n                except KeyError:\n                    pass\n                '
                img = bpy.data.images.load(filepath)
                (w, h) = img.size
                cam['imageWidth'] = w
                cam['imageHeight'] = h
                try:
                    cam['orientation'] = exif['Orientation']
                except KeyError:
                    cam['orientation'] = 1
                if cam['orientation'] == 8:
                    cam_obj.rotation_euler[1] -= pi / 2
                if cam['orientation'] == 6:
                    cam_obj.rotation_euler[1] += pi / 2
                if cam['orientation'] == 3:
                    cam_obj.rotation_euler[1] += pi
                if scn.camera is None:
                    bpy.ops.camera.geophotos_setactive('EXEC_DEFAULT', camLst=cam_obj.name)
        return {'FINISHED'}

class CAMERA_OT_geophotos_setactive(Operator):
    bl_idname = 'camera.geophotos_setactive'
    bl_description = 'Switch active geophoto camera'
    bl_label = 'Switch geophoto camera'
    bl_options = {'REGISTER'}

    def listGeoCam(self, context):
        if False:
            print('Hello World!')
        scn = context.scene
        return [(obj.name, obj.name, obj.name) for obj in scn.objects if obj.type == 'CAMERA' and 'background' in obj.data]
    camLst: EnumProperty(name='Camera', description='Select camera', items=listGeoCam)

    def draw(self, context):
        if False:
            i = 10
            return i + 15
        layout = self.layout
        layout.prop(self, 'camLst')

    def invoke(self, context, event):
        if False:
            return 10
        if len(self.camLst) == 0:
            self.report({'ERROR'}, 'No valid camera')
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if False:
            print('Hello World!')
        if context.space_data.type != 'VIEW_3D':
            self.report({'ERROR'}, 'Wrong context')
            return {'CANCELLED'}
        scn = context.scene
        view3d = context.space_data
        cam_obj = scn.objects[self.camLst]
        cam_obj.select_set(True)
        context.view_layer.objects.active = cam_obj
        cam = cam_obj.data
        scn.camera = cam_obj
        scn.render.resolution_x = cam['imageWidth']
        scn.render.resolution_y = cam['imageHeight']
        scn.render.resolution_percentage = 100
        filepath = cam['background']
        try:
            img = [img for img in bpy.data.images if img.filepath == filepath][0]
        except IndexError:
            img = bpy.data.images.load(filepath)
        cam.show_background_images = True
        for bkg in cam.background_images:
            bkg.show_background_image = False
        bkgs = [bkg for bkg in cam.background_images if bkg.image is not None]
        try:
            bkg = [bkg for bkg in bkgs if bkg.image.filepath == filepath][0]
        except IndexError:
            bkg = cam.background_images.new()
            bkg.image = img
        bkg.show_background_image = True
        bkg.alpha = 1
        return {'FINISHED'}
classes = [CAMERA_OT_geophotos_add, CAMERA_OT_geophotos_setactive]

def register():
    if False:
        while True:
            i = 10
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            log.warning('{} is already registered, now unregister and retry... '.format(cls))
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)

def unregister():
    if False:
        print('Hello World!')
    for cls in classes:
        bpy.utils.unregister_class(cls)