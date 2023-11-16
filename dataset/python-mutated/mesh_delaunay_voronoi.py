import bpy
import time
from .utils import computeVoronoiDiagram, computeDelaunayTriangulation
from ..core.utils import perf_clock
try:
    from mathutils.geometry import delaunay_2d_cdt
except ImportError:
    NATIVE = False
else:
    NATIVE = True
import logging
log = logging.getLogger(__name__)

class Point:

    def __init__(self, x, y, z):
        if False:
            for i in range(10):
                print('nop')
        (self.x, self.y, self.z) = (x, y, z)

def unique(L):
    if False:
        i = 10
        return i + 15
    'Return a list of unhashable elements in s, but without duplicates.\n\t[[1, 2], [2, 3], [1, 2]] >>> [[1, 2], [2, 3]]'
    nDupli = 0
    nZcolinear = 0
    L.sort()
    last = L[-1]
    for i in range(len(L) - 2, -1, -1):
        if last[:2] == L[i][:2]:
            if last[2] == L[i][2]:
                nDupli += 1
            else:
                nZcolinear += 1
            del L[i]
        else:
            last = L[i]
    return (nDupli, nZcolinear)

def checkEqual(lst):
    if False:
        for i in range(10):
            print('nop')
    return lst[1:] == lst[:-1]

class OBJECT_OT_tesselation_delaunay(bpy.types.Operator):
    bl_idname = 'tesselation.delaunay'
    bl_label = 'Triangulation'
    bl_description = 'Terrain points cloud Delaunay triangulation in 2.5D'
    bl_options = {'UNDO'}

    def execute(self, context):
        if False:
            print('Hello World!')
        w = context.window
        w.cursor_set('WAIT')
        t0 = perf_clock()
        objs = context.selected_objects
        if len(objs) == 0 or len(objs) > 1:
            self.report({'INFO'}, 'Selection is empty or too much object selected')
            return {'CANCELLED'}
        obj = objs[0]
        if obj.type != 'MESH':
            self.report({'INFO'}, "Selection isn't a mesh")
            return {'CANCELLED'}
        r = obj.rotation_euler
        s = obj.scale
        mesh = obj.data
        if NATIVE:
            '\n\t\t\tUse native Delaunay triangulation function : delaunay_2d_cdt(verts, edges, faces, output_type, epsilon) >> [verts, edges, faces, orig_verts, orig_edges, orig_faces]\n\t\t\tThe three returned orig lists give, for each of verts, edges, and faces, the list of input element indices corresponding to the positionally same output element. For edges, the orig indices start with the input edges and then continue with the edges implied by each of the faces (n of them for an n-gon).\n\t\t\tOutput type :\n\t\t\t# 0 => triangles with convex hull.\n\t\t\t# 1 => triangles inside constraints.\n\t\t\t# 2 => the input constraints, intersected.\n\t\t\t# 3 => like 2 but with extra edges to make valid BMesh faces.\n\t\t\t'
            log.info('Triangulate {} points...'.format(len(mesh.vertices)))
            (verts, edges, faces, overts, oedges, ofaces) = delaunay_2d_cdt([v.co.to_2d() for v in mesh.vertices], [], [], 0, 0.1)
            verts = [(v.x, v.y, mesh.vertices[overts[i][0]].co.z) for (i, v) in enumerate(verts)]
            log.info('Getting {} triangles'.format(len(faces)))
            log.info('Create mesh...')
            tinMesh = bpy.data.meshes.new('TIN')
            tinMesh.from_pydata(verts, edges, faces)
            tinMesh.update()
        else:
            vertsPts = [vertex.co for vertex in mesh.vertices]
            verts = [[vert.x, vert.y, vert.z] for vert in vertsPts]
            (nDupli, nZcolinear) = unique(verts)
            nVerts = len(verts)
            log.info('{} duplicates points ignored'.format(nDupli))
            log.info('{} z colinear points excluded'.format(nZcolinear))
            if nVerts < 3:
                self.report({'ERROR'}, 'Not enough points')
                return {'CANCELLED'}
            xValues = [pt[0] for pt in verts]
            yValues = [pt[1] for pt in verts]
            if checkEqual(xValues) or checkEqual(yValues):
                self.report({'ERROR'}, 'Points are colinear')
                return {'CANCELLED'}
            log.info('Triangulate {} points...'.format(nVerts))
            vertsPts = [Point(vert[0], vert[1], vert[2]) for vert in verts]
            faces = computeDelaunayTriangulation(vertsPts)
            faces = [tuple(reversed(tri)) for tri in faces]
            log.info('Getting {} triangles'.format(len(faces)))
            log.info('Create mesh...')
            tinMesh = bpy.data.meshes.new('TIN')
            tinMesh.from_pydata(verts, [], faces)
            tinMesh.update(calc_edges=True)
        tinObj = bpy.data.objects.new('TIN', tinMesh)
        tinObj.location = obj.location.copy()
        tinObj.rotation_euler = r
        tinObj.scale = s
        context.scene.collection.objects.link(tinObj)
        context.view_layer.objects.active = tinObj
        tinObj.select_set(True)
        obj.select_set(False)
        t = round(perf_clock() - t0, 2)
        msg = '{} triangles created in {} seconds'.format(len(faces), t)
        self.report({'INFO'}, msg)
        return {'FINISHED'}

class OBJECT_OT_tesselation_voronoi(bpy.types.Operator):
    bl_idname = 'tesselation.voronoi'
    bl_label = 'Diagram'
    bl_description = 'Points cloud Voronoi diagram in 2D'
    bl_options = {'REGISTER', 'UNDO'}
    meshType: bpy.props.EnumProperty(items=[('Edges', 'Edges', ''), ('Faces', 'Faces', '')], name='Mesh type', description='')
    '\n\tdef draw(self, context):\n\t'

    def execute(self, context):
        if False:
            while True:
                i = 10
        w = context.window
        w.cursor_set('WAIT')
        t0 = perf_clock()
        objs = context.selected_objects
        if len(objs) == 0 or len(objs) > 1:
            self.report({'INFO'}, 'Selection is empty or too much object selected')
            return {'CANCELLED'}
        obj = objs[0]
        if obj.type != 'MESH':
            self.report({'INFO'}, "Selection isn't a mesh")
            return {'CANCELLED'}
        r = obj.rotation_euler
        s = obj.scale
        mesh = obj.data
        vertsPts = [vertex.co for vertex in mesh.vertices]
        verts = [[vert.x, vert.y, vert.z] for vert in vertsPts]
        (nDupli, nZcolinear) = unique(verts)
        nVerts = len(verts)
        log.info('{} duplicates points ignored'.format(nDupli))
        log.info('{} z colinear points excluded'.format(nZcolinear))
        if nVerts < 3:
            self.report({'ERROR'}, 'Not enough points')
            return {'CANCELLED'}
        xValues = [pt[0] for pt in verts]
        yValues = [pt[1] for pt in verts]
        if checkEqual(xValues) or checkEqual(yValues):
            self.report({'ERROR'}, 'Points are colinear')
            return {'CANCELLED'}
        log.info('Tesselation... ({} points)'.format(nVerts))
        (xbuff, ybuff) = (5, 5)
        zPosition = 0
        vertsPts = [Point(vert[0], vert[1], vert[2]) for vert in verts]
        if self.meshType == 'Edges':
            (pts, edgesIdx) = computeVoronoiDiagram(vertsPts, xbuff, ybuff, polygonsOutput=False, formatOutput=True)
        else:
            (pts, polyIdx) = computeVoronoiDiagram(vertsPts, xbuff, ybuff, polygonsOutput=True, formatOutput=True, closePoly=False)
        pts = [[pt[0], pt[1], zPosition] for pt in pts]
        log.info('Create mesh...')
        voronoiDiagram = bpy.data.meshes.new('VoronoiDiagram')
        if self.meshType == 'Edges':
            voronoiDiagram.from_pydata(pts, edgesIdx, [])
        else:
            voronoiDiagram.from_pydata(pts, [], list(polyIdx.values()))
        voronoiDiagram.update(calc_edges=True)
        voronoiObj = bpy.data.objects.new('VoronoiDiagram', voronoiDiagram)
        voronoiObj.location = obj.location.copy()
        voronoiObj.rotation_euler = r
        voronoiObj.scale = s
        context.scene.collection.objects.link(voronoiObj)
        context.view_layer.objects.active = voronoiObj
        voronoiObj.select_set(True)
        obj.select_set(False)
        t = round(perf_clock() - t0, 2)
        if self.meshType == 'Edges':
            self.report({'INFO'}, '{} edges created in {} seconds'.format(len(edgesIdx), t))
        else:
            self.report({'INFO'}, '{} polygons created in {} seconds'.format(len(polyIdx), t))
        return {'FINISHED'}
classes = [OBJECT_OT_tesselation_delaunay, OBJECT_OT_tesselation_voronoi]

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