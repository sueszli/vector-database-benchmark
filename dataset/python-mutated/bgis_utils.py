import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras.view3d_utils import region_2d_to_location_3d, region_2d_to_vector_3d
from ...core import BBOX

def isTopView(context):
    if False:
        for i in range(10):
            print('nop')
    if context.area.type == 'VIEW_3D':
        reg3d = context.region_data
    else:
        return False
    return reg3d.view_perspective == 'ORTHO' and tuple(reg3d.view_matrix.to_euler()) == (0, 0, 0)

def mouseTo3d(context, x, y):
    if False:
        i = 10
        return i + 15
    'Convert event.mouse_region to world coordinates'
    if context.area.type != 'VIEW_3D':
        raise Exception('Wrong context')
    coords = (x, y)
    reg = context.region
    reg3d = context.region_data
    vec = region_2d_to_vector_3d(reg, reg3d, coords)
    loc = region_2d_to_location_3d(reg, reg3d, coords, vec)
    return loc

class DropToGround:
    """A class to perform raycasting accross z axis"""

    def __init__(self, scn, ground, method='OBJ'):
        if False:
            i = 10
            return i + 15
        self.method = method
        self.scn = scn
        self.ground = ground
        self.bbox = getBBOX.fromObj(ground, applyTransform=True)
        self.mw = self.ground.matrix_world
        self.mwi = self.mw.inverted()
        if self.method == 'BVH':
            self.bvh = BVHTree.FromObject(self.ground, bpy.context.evaluated_depsgraph_get(), deform=True)

    def rayCast(self, x, y):
        if False:
            return 10
        offset = 100
        orgWldSpace = Vector((x, y, self.bbox.zmax + offset))
        orgObjSpace = self.mwi @ orgWldSpace
        direction = Vector((0, 0, -1))

        class RayCastHit:
            pass
        rcHit = RayCastHit()
        if self.method == 'OBJ':
            (rcHit.hit, rcHit.loc, rcHit.normal, rcHit.faceIdx) = self.ground.ray_cast(orgObjSpace, direction)
        elif self.method == 'BVH':
            (rcHit.loc, rcHit.normal, rcHit.faceIdx, rcHit.dst) = self.bvh.ray_cast(orgObjSpace, direction)
            if not rcHit.loc:
                rcHit.hit = False
            else:
                rcHit.hit = True
        if not rcHit.hit:
            rcHit.loc = Vector((orgWldSpace.x, orgWldSpace.y, 0))
        else:
            rcHit.hit = True
        rcHit.loc = self.mw @ rcHit.loc
        return rcHit

def placeObj(mesh, objName):
    if False:
        return 10
    'Build and add a new object from a given mesh'
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects.new(objName, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj

def adjust3Dview(context, bbox, zoomToSelect=True):
    if False:
        print('Hello World!')
    'adjust all 3d views clip distance to match the submited bbox'
    dst = round(max(bbox.dimensions))
    k = 5
    dst = dst * k
    areas = context.screen.areas
    for area in areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            if dst < 100:
                space.clip_start = 1
            elif dst < 1000:
                space.clip_start = 10
            else:
                space.clip_start = 100
            if space.clip_end < dst:
                if dst > 10000000:
                    dst = 10000000
                space.clip_end = dst
            if zoomToSelect:
                overrideContext = context.copy()
                overrideContext['area'] = area
                overrideContext['region'] = area.regions[-1]
                bpy.ops.view3d.view_selected(overrideContext)

def showTextures(context):
    if False:
        i = 10
        return i + 15
    'Force view mode with textures'
    scn = context.scene
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            if space.shading.type == 'SOLID':
                space.shading.color_type = 'TEXTURE'

def addTexture(mat, img, uvLay, name='texture'):
    if False:
        print('Hello World!')
    'Set a new image texture to a given material and following a given uv map'
    engine = bpy.context.scene.render.engine
    mat.use_nodes = True
    node_tree = mat.node_tree
    node_tree.nodes.clear()
    uvMapNode = node_tree.nodes.new('ShaderNodeUVMap')
    uvMapNode.uv_map = uvLay.name
    uvMapNode.location = (-800, 200)
    textureNode = node_tree.nodes.new('ShaderNodeTexImage')
    textureNode.image = img
    textureNode.extension = 'CLIP'
    textureNode.show_texture = True
    textureNode.location = (-400, 200)
    diffuseNode = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    diffuseNode.location = (0, 200)
    outputNode = node_tree.nodes.new('ShaderNodeOutputMaterial')
    outputNode.location = (400, 200)
    node_tree.links.new(uvMapNode.outputs['UV'], textureNode.inputs['Vector'])
    node_tree.links.new(textureNode.outputs['Color'], diffuseNode.inputs['Base Color'])
    node_tree.links.new(diffuseNode.outputs['BSDF'], outputNode.inputs['Surface'])

class getBBOX:
    """Utilities to build BBOX object from various Blender context"""

    @staticmethod
    def fromObj(obj, applyTransform=True):
        if False:
            print('Hello World!')
        'Create a 3D BBOX from Blender object'
        if applyTransform:
            boundPts = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        else:
            boundPts = obj.bound_box
        xmin = min([pt[0] for pt in boundPts])
        xmax = max([pt[0] for pt in boundPts])
        ymin = min([pt[1] for pt in boundPts])
        ymax = max([pt[1] for pt in boundPts])
        zmin = min([pt[2] for pt in boundPts])
        zmax = max([pt[2] for pt in boundPts])
        return BBOX(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

    @classmethod
    def fromScn(cls, scn):
        if False:
            print('Hello World!')
        'Create a 3D BBOX from Blender Scene\n\t\tunion of bounding box of all objects containing in the scene'
        objs = [obj for obj in scn.collection.all_objects if obj.empty_display_type != 'IMAGE']
        if len(objs) == 0:
            scnBbox = BBOX(0, 0, 0, 0, 0, 0)
        else:
            scnBbox = cls.fromObj(objs[0])
        for obj in objs:
            bbox = cls.fromObj(obj)
            scnBbox += bbox
        return scnBbox

    @staticmethod
    def fromBmesh(bm):
        if False:
            i = 10
            return i + 15
        'Create a 3D bounding box from a bmesh object'
        xmin = min([pt.co.x for pt in bm.verts])
        xmax = max([pt.co.x for pt in bm.verts])
        ymin = min([pt.co.y for pt in bm.verts])
        ymax = max([pt.co.y for pt in bm.verts])
        zmin = min([pt.co.z for pt in bm.verts])
        zmax = max([pt.co.z for pt in bm.verts])
        return BBOX(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

    @staticmethod
    def fromTopView(context):
        if False:
            print('Hello World!')
        'Create a 2D BBOX from Blender 3dview if the view is top left ortho else return None'
        scn = context.scene
        area = context.area
        if area.type != 'VIEW_3D':
            return None
        reg = context.region
        reg3d = context.region_data
        if reg3d.view_perspective != 'ORTHO' or tuple(reg3d.view_matrix.to_euler()) != (0, 0, 0):
            print('View3d must be in top ortho')
            return None
        loc = mouseTo3d(context, area.width, area.height)
        (xmax, ymax) = (loc.x, loc.y)
        loc = mouseTo3d(context, 0, 0)
        (xmin, ymin) = (loc.x, loc.y)
        return BBOX(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)