import logging
log = logging.getLogger(__name__)
import bpy
import bmesh
from .utils import DropToGround, getBBOX
from mathutils import Vector, Matrix
from bpy.types import Operator
from bpy.props import BoolProperty, EnumProperty

def get_align_matrix(location, normal):
    if False:
        print('Hello World!')
    up = Vector((0, 0, 1))
    angle = normal.angle(up)
    axis = up.cross(normal)
    mat_rot = Matrix.Rotation(angle, 4, axis)
    mat_loc = Matrix.Translation(location)
    mat_align = mat_rot @ mat_loc
    return mat_align

def get_lowest_world_co(ob, mat_parent=None):
    if False:
        print('Hello World!')
    bme = bmesh.new()
    bme.from_mesh(ob.data)
    mat_to_world = ob.matrix_world.copy()
    if mat_parent:
        mat_to_world = mat_parent @ mat_to_world
    lowest = None
    for v in bme.verts:
        if not lowest:
            lowest = v
        if (mat_to_world @ v.co).z < (mat_to_world @ lowest.co).z:
            lowest = v
    lowest_co = mat_to_world @ lowest.co
    bme.free()
    return lowest_co

class OBJECT_OT_drop_to_ground(Operator):
    bl_idname = 'object.drop'
    bl_label = 'Drop to Ground'
    bl_description = 'Drop selected objects on the Active object'
    bl_options = {'REGISTER', 'UNDO'}
    align: BoolProperty(name='Align to ground', description="Aligns the objects' rotation to the ground", default=False)
    axisAlign: EnumProperty(items=[('N', 'Normal', 'Ground normal'), ('X', 'X', 'Ground X normal'), ('Y', 'Y', 'Ground Y normal'), ('Z', 'Z', 'Ground Z normal')], name='Align axis', description='')
    useOrigin: BoolProperty(name='Use Origins', description="Drop to objects' origins\nUse this option for dropping all types of Objects", default=False)

    @classmethod
    def poll(cls, context):
        if False:
            i = 10
            return i + 15
        act_obj = context.active_object
        return context.mode == 'OBJECT' and len(context.selected_objects) >= 2 and act_obj and (act_obj.type in {'MESH', 'FONT', 'META', 'CURVE', 'SURFACE'})

    def draw(self, context):
        if False:
            return 10
        layout = self.layout
        layout.prop(self, 'align')
        if self.align:
            layout.prop(self, 'axisAlign')
        layout.prop(self, 'useOrigin')

    def execute(self, context):
        if False:
            return 10
        bpy.context.view_layer.update()
        ground = context.active_object
        obs = context.selected_objects
        if ground in obs:
            obs.remove(ground)
        scn = context.scene
        rayCaster = DropToGround(scn, ground)
        for ob in obs:
            if self.useOrigin:
                minLoc = ob.location
            else:
                minLoc = get_lowest_world_co(ob)
            if not minLoc:
                msg = 'Object {} is of type {} works only with Use Center option checked'.format(ob.name, ob.type)
                log.info(msg)
            (x, y) = (minLoc.x, minLoc.y)
            hit = rayCaster.rayCast(x, y)
            if not hit.hit:
                log.info(ob.name + ' did not hit the Active Object')
                continue
            down = hit.loc - minLoc
            ob.location += down
            if self.align:
                vect = ob.location - hit.loc
                normal = get_align_matrix(hit.loc, hit.normal)
                rot = normal.to_euler()
                if self.axisAlign == 'X':
                    rot.y = 0
                    rot.z = 0
                elif self.axisAlign == 'Y':
                    rot.x = 0
                    rot.z = 0
                elif self.axisAlign == 'Z':
                    rot.x = 0
                    rot.y = 0
                matrix = ob.matrix_world.copy().to_3x3()
                matrix.rotate(rot)
                matrix = matrix.to_4x4()
                ob.matrix_world = matrix
                ob.location = hit.loc
                vect.rotate(rot)
                ob.location += vect
        return {'FINISHED'}

def register():
    if False:
        i = 10
        return i + 15
    try:
        bpy.utils.register_class(OBJECT_OT_drop_to_ground)
    except ValueError as e:
        log.warning('{} is already registered, now unregister and retry... '.format(OBJECT_OT_drop_to_ground))
        unregister()
        bpy.utils.register_class(OBJECT_OT_drop_to_ground)

def unregister():
    if False:
        while True:
            i = 10
    bpy.utils.unregister_class(OBJECT_OT_drop_to_ground)