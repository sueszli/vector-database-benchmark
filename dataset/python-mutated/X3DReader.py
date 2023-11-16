from math import pi, sin, cos, sqrt
from typing import Dict
import numpy
from UM.Job import Job
from UM.Logger import Logger
from UM.Math.Matrix import Matrix
from UM.Math.Vector import Vector
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Mesh.MeshReader import MeshReader
from cura.Scene.CuraSceneNode import CuraSceneNode as SceneNode
MYPY = False
try:
    if not MYPY:
        import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
DEFAULT_SUBDIV = 16
EPSILON = 1e-06

class Shape:

    def __init__(self, verts, faces, index_base, name):
        if False:
            while True:
                i = 10
        self.verts = verts
        self.faces = faces
        self.index_base = index_base
        self.name = name

class X3DReader(MeshReader):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._supported_extensions = ['.x3d']
        self._namespaces = {}

    def _read(self, file_name):
        if False:
            i = 10
            return i + 15
        try:
            self.defs = {}
            self.shapes = []
            tree = ET.parse(file_name)
            xml_root = tree.getroot()
            if xml_root.tag != 'X3D':
                return None
            scale = 1000
            if xml_root[0].tag == 'head':
                for head_node in xml_root[0]:
                    if head_node.tag == 'unit' and head_node.attrib.get('category') == 'length':
                        scale *= float(head_node.attrib['conversionFactor'])
                        break
                xml_scene = xml_root[1]
            else:
                xml_scene = xml_root[0]
            if xml_scene.tag != 'Scene':
                return None
            self.transform = Matrix()
            self.transform.setByScaleFactor(scale)
            self.index_base = 0
            self.processChildNodes(xml_scene)
            if self.shapes:
                builder = MeshBuilder()
                builder.setVertices(numpy.concatenate([shape.verts for shape in self.shapes]))
                builder.setIndices(numpy.concatenate([shape.faces for shape in self.shapes]))
                builder.calculateNormals()
                builder.setFileName(file_name)
                mesh_data = builder.build()
                mesh_data.getExtents()
                node = SceneNode()
                node.setMeshData(mesh_data)
                node.setSelectable(True)
                node.setName(file_name)
            else:
                return None
        except Exception:
            Logger.logException('e', 'Exception in X3D reader')
            return None
        return node

    def processNode(self, xml_node):
        if False:
            i = 10
            return i + 15
        xml_node = self.resolveDefUse(xml_node)
        if xml_node is None:
            return
        tag = xml_node.tag
        if tag in ('Group', 'StaticGroup', 'CADAssembly', 'CADFace', 'CADLayer', 'Collision'):
            self.processChildNodes(xml_node)
        if tag == 'CADPart':
            self.processTransform(xml_node)
        elif tag == 'LOD':
            self.processNode(xml_node[0])
        elif tag == 'Transform':
            self.processTransform(xml_node)
        elif tag == 'Shape':
            self.processShape(xml_node)

    def processShape(self, xml_node):
        if False:
            i = 10
            return i + 15
        geometry = appearance = None
        for sub_node in xml_node:
            if sub_node.tag == 'Appearance' and (not appearance):
                appearance = self.resolveDefUse(sub_node)
            elif sub_node.tag in self.geometry_importers and (not geometry):
                geometry = self.resolveDefUse(sub_node)
        if geometry is not None:
            try:
                self.verts = self.faces = []
                self.geometry_importers[geometry.tag](self, geometry)
                m = self.transform.getData()
                verts = m.dot(self.verts)[:3].transpose()
                self.shapes.append(Shape(verts, self.faces, self.index_base, geometry.tag))
                self.index_base += len(verts)
            except Exception:
                Logger.logException('e', 'Exception in X3D reader while reading %s', geometry.tag)

    def resolveDefUse(self, node):
        if False:
            print('Hello World!')
        USE = node.attrib.get('USE')
        if USE:
            return self.defs.get(USE, None)
        DEF = node.attrib.get('DEF')
        if DEF:
            self.defs[DEF] = node
        return node

    def processChildNodes(self, node):
        if False:
            while True:
                i = 10
        for c in node:
            self.processNode(c)
            Job.yieldThread()

    def processTransform(self, node):
        if False:
            while True:
                i = 10
        rot = readRotation(node, 'rotation', (0, 0, 1, 0))
        trans = readVector(node, 'translation', (0, 0, 0))
        scale = readVector(node, 'scale', (1, 1, 1))
        center = readVector(node, 'center', (0, 0, 0))
        scale_orient = readRotation(node, 'scaleOrientation', (0, 0, 1, 0))
        prev = Matrix(self.transform.getData())
        got_center = center.x != 0 or center.y != 0 or center.z != 0
        T = self.transform
        if trans.x != 0 or trans.y != 0 or trans.z != 0:
            T.translate(trans)
        if got_center:
            T.translate(center)
        if rot[0] != 0:
            T.rotateByAxis(*rot)
        if scale.x != 1 or scale.y != 1 or scale.z != 1:
            got_scale_orient = scale_orient[0] != 0
            if got_scale_orient:
                T.rotateByAxis(*scale_orient)
            S = Matrix()
            S.setByScaleVector(scale)
            T.multiply(S)
            if got_scale_orient:
                T.rotateByAxis(-scale_orient[0], scale_orient[1])
        if got_center:
            T.translate(-center)
        self.processChildNodes(node)
        self.transform = prev

    def processGeometryBox(self, node):
        if False:
            while True:
                i = 10
        (dx, dy, dz) = readFloatArray(node, 'size', [2, 2, 2])
        dx /= 2
        dy /= 2
        dz /= 2
        self.reserveFaceAndVertexCount(12, 8)
        self.addVertex(dx, dy, dz)
        self.addVertex(-dx, dy, dz)
        self.addVertex(-dx, dy, -dz)
        self.addVertex(dx, dy, -dz)
        self.addVertex(dx, -dy, dz)
        self.addVertex(-dx, -dy, dz)
        self.addVertex(-dx, -dy, -dz)
        self.addVertex(dx, -dy, -dz)
        self.addQuad(0, 1, 2, 3)
        self.addQuad(4, 0, 3, 7)
        self.addQuad(7, 3, 2, 6)
        self.addQuad(6, 2, 1, 5)
        self.addQuad(5, 1, 0, 4)
        self.addQuad(7, 6, 5, 4)

    def processGeometrySphere(self, node):
        if False:
            return 10
        r = readFloat(node, 'radius', 0.5)
        subdiv = readIntArray(node, 'subdivision', None)
        if subdiv:
            if len(subdiv) == 1:
                nr = ns = subdiv[0]
            else:
                (nr, ns) = subdiv
        else:
            nr = ns = DEFAULT_SUBDIV
        lau = pi / nr
        lou = 2 * pi / ns
        self.reserveFaceAndVertexCount(ns * (nr * 2 - 2), 2 + (nr - 1) * ns)
        self.addVertex(0, r, 0)
        self.addVertex(0, -r, 0)
        for ring in range(1, nr):
            for seg in range(ns):
                self.addVertex(-r * sin(lou * seg) * sin(lau * ring), r * cos(lau * ring), -r * cos(lou * seg) * sin(lau * ring))
        vb = 2 + (nr - 2) * ns
        for seg in range(ns):
            self.addTri(0, seg + 2, (seg + 1) % ns + 2)
            self.addTri(1, vb + (seg + 1) % ns, vb + seg)
        for ring in range(nr - 2):
            tvb = 2 + ring * ns
            bvb = tvb + ns
            for seg in range(ns):
                nseg = (seg + 1) % ns
                self.addQuad(tvb + seg, bvb + seg, bvb + nseg, tvb + nseg)

    def processGeometryCone(self, node):
        if False:
            print('Hello World!')
        r = readFloat(node, 'bottomRadius', 1)
        height = readFloat(node, 'height', 2)
        bottom = readBoolean(node, 'bottom', True)
        side = readBoolean(node, 'side', True)
        n = readInt(node, 'subdivision', DEFAULT_SUBDIV)
        d = height / 2
        angle = 2 * pi / n
        self.reserveFaceAndVertexCount((n if side else 0) + (n - 2 if bottom else 0), n + 1)
        self.addVertex(0, d, 0)
        for i in range(n):
            self.addVertex(-r * sin(angle * i), -d, -r * cos(angle * i))
        if side:
            for i in range(n):
                self.addTri(1 + (i + 1) % n, 0, 1 + i)
        if bottom:
            for i in range(2, n):
                self.addTri(1, i, i + 1)

    def processGeometryCylinder(self, node):
        if False:
            while True:
                i = 10
        r = readFloat(node, 'radius', 1)
        height = readFloat(node, 'height', 2)
        bottom = readBoolean(node, 'bottom', True)
        side = readBoolean(node, 'side', True)
        top = readBoolean(node, 'top', True)
        n = readInt(node, 'subdivision', DEFAULT_SUBDIV)
        nn = n * 2
        angle = 2 * pi / n
        hh = height / 2
        self.reserveFaceAndVertexCount((nn if side else 0) + (n - 2 if top else 0) + (n - 2 if bottom else 0), nn)
        for i in range(n):
            rs = -r * sin(angle * i)
            rc = -r * cos(angle * i)
            self.addVertex(rs, hh, rc)
            self.addVertex(rs, -hh, rc)
        if side:
            for i in range(n):
                ni = (i + 1) % n
                self.addQuad(ni * 2 + 1, ni * 2, i * 2, i * 2 + 1)
        for i in range(2, nn - 3, 2):
            if top:
                self.addTri(0, i, i + 2)
            if bottom:
                self.addTri(1, i + 1, i + 3)

    def processGeometryElevationGrid(self, node):
        if False:
            i = 10
            return i + 15
        dx = readFloat(node, 'xSpacing', 1)
        dz = readFloat(node, 'zSpacing', 1)
        nx = readInt(node, 'xDimension', 0)
        nz = readInt(node, 'zDimension', 0)
        height = readFloatArray(node, 'height', False)
        ccw = readBoolean(node, 'ccw', True)
        if nx <= 0 or nz <= 0 or len(height) < nx * nz:
            return
        self.reserveFaceAndVertexCount(2 * (nx - 1) * (nz - 1), nx * nz)
        for z in range(nz):
            for x in range(nx):
                self.addVertex(x * dx, height[z * nx + x], z * dz)
        for z in range(1, nz):
            for x in range(1, nx):
                self.addTriFlip((z - 1) * nx + x - 1, z * nx + x, (z - 1) * nx + x, ccw)
                self.addTriFlip((z - 1) * nx + x - 1, z * nx + x - 1, z * nx + x, ccw)

    def processGeometryExtrusion(self, node):
        if False:
            i = 10
            return i + 15
        ccw = readBoolean(node, 'ccw', True)
        begin_cap = readBoolean(node, 'beginCap', True)
        end_cap = readBoolean(node, 'endCap', True)
        cross = readFloatArray(node, 'crossSection', (1, 1, 1, -1, -1, -1, -1, 1, 1, 1))
        cross = [(cross[i], cross[i + 1]) for i in range(0, len(cross), 2)]
        spine = readFloatArray(node, 'spine', (0, 0, 0, 0, 1, 0))
        spine = [(spine[i], spine[i + 1], spine[i + 2]) for i in range(0, len(spine), 3)]
        orient = readFloatArray(node, 'orientation', None)
        if orient:

            def toRotationMatrix(rot):
                if False:
                    return 10
                (x, y, z) = rot[:3]
                a = rot[3]
                s = sin(a)
                c = cos(a)
                t = 1 - c
                return numpy.array(((x * x * t + c, x * y * t - z * s, x * z * t + y * s), (x * y * t + z * s, y * y * t + c, y * z * t - x * s), (x * z * t - y * s, y * z * t + x * s, z * z * t + c)))
            orient = [toRotationMatrix(orient[i:i + 4]) if orient[i + 3] != 0 else None for i in range(0, len(orient), 4)]
        scale = readFloatArray(node, 'scale', None)
        if scale:
            scale = [numpy.array(((scale[i], 0, 0), (0, 1, 0), (0, 0, scale[i + 1]))) if scale[i] != 1 or scale[i + 1] != 1 else None for i in range(0, len(scale), 2)]
        crossClosed = cross[0] == cross[-1]
        if crossClosed:
            cross = cross[:-1]
        nc = len(cross)
        cross = [numpy.array((c[0], 0, c[1])) for c in cross]
        ncf = nc if crossClosed else nc - 1
        spine_closed = spine[0] == spine[-1]
        if spine_closed:
            spine = spine[:-1]
        ns = len(spine)
        spine = [Vector(*s) for s in spine]
        nsf = ns if spine_closed else ns - 1

        def findFirstAngleNormal():
            if False:
                i = 10
                return i + 15
            for i in range(1, ns - 1):
                spt = spine[i]
                z = (spine[i + 1] - spt).cross(spine[i - 1] - spt)
                if z.length() > EPSILON:
                    return z
            if len(spine) < 2:
                return Vector(0, 0, 1)
            v = spine[1] - spine[0]
            orig_y = Vector(0, 1, 0)
            orig_z = Vector(0, 0, 1)
            if v.cross(orig_y).length() > EPSILON:
                a = v.cross(orig_y)
                (x, y, z) = a.normalized().getData()
                s = a.length() / v.length()
                c = sqrt(1 - s * s)
                t = 1 - c
                m = numpy.array(((x * x * t + c, x * y * t + z * s, x * z * t - y * s), (x * y * t - z * s, y * y * t + c, y * z * t + x * s), (x * z * t + y * s, y * z * t - x * s, z * z * t + c)))
                orig_z = Vector(*m.dot(orig_z.getData()))
            return orig_z
        self.reserveFaceAndVertexCount(2 * nsf * ncf + (nc - 2 if begin_cap else 0) + (nc - 2 if end_cap else 0), ns * nc)
        z = None
        for (i, spt) in enumerate(spine):
            if i > 0 and i < ns - 1 or spine_closed:
                snext = spine[(i + 1) % ns]
                sprev = spine[(i - 1 + ns) % ns]
                y = snext - sprev
                vnext = snext - spt
                vprev = sprev - spt
                try_z = vnext.cross(vprev)
                if try_z.length() > EPSILON:
                    if z is not None and try_z.dot(z) < 0:
                        try_z = -try_z
                    z = try_z
                elif not z:
                    z = findFirstAngleNormal()
            elif i == 0:
                snext = spine[i + 1]
                y = snext - spt
                z = findFirstAngleNormal()
            else:
                sprev = spine[i - 1]
                y = spt - sprev
            z = z.normalized()
            y = y.normalized()
            x = y.cross(z)
            m = numpy.array(((x.x, y.x, z.x), (x.y, y.y, z.y), (x.z, y.z, z.z)))
            if orient:
                mrot = orient[i] if len(orient) > 1 else orient[0]
                if mrot is not None:
                    m = m.dot(mrot)
            if scale:
                mscale = scale[i] if len(scale) > 1 else scale[0]
                if mscale is not None:
                    m = m.dot(mscale)
            sptv3 = numpy.array(spt.getData()[:3])
            for cpt in cross:
                v = sptv3 + m.dot(cpt)
                self.addVertex(*v)
        if begin_cap:
            self.addFace([x for x in range(nc - 1, -1, -1)], ccw)
        for s in range(ns - 1):
            for c in range(ncf):
                self.addQuadFlip(s * nc + c, s * nc + (c + 1) % nc, (s + 1) * nc + (c + 1) % nc, (s + 1) * nc + c, ccw)
        if spine_closed:
            b = (ns - 1) * nc
            for c in range(ncf):
                self.addQuadFlip(b + c, b + (c + 1) % nc, (c + 1) % nc, c, ccw)
        if end_cap:
            self.addFace([(ns - 1) * nc + x for x in range(0, nc)], ccw)

    def startCoordMesh(self, node, num_faces):
        if False:
            while True:
                i = 10
        ccw = readBoolean(node, 'ccw', True)
        self.readVertices(node)
        if hasattr(num_faces, '__call__'):
            num_faces = num_faces(self.getVertexCount())
        self.reserveFaceCount(num_faces)
        return ccw

    def processGeometryIndexedTriangleSet(self, node):
        if False:
            return 10
        index = readIntArray(node, 'index', [])
        num_faces = len(index) // 3
        ccw = int(self.startCoordMesh(node, num_faces))
        for i in range(0, num_faces * 3, 3):
            self.addTri(index[i + 1 - ccw], index[i + ccw], index[i + 2])

    def processGeometryIndexedTriangleStripSet(self, node):
        if False:
            while True:
                i = 10
        strips = readIndex(node, 'index')
        ccw = int(self.startCoordMesh(node, sum([len(strip) - 2 for strip in strips])))
        for strip in strips:
            sccw = ccw
            for i in range(len(strip) - 2):
                self.addTri(strip[i + 1 - sccw], strip[i + sccw], strip[i + 2])
                sccw = 1 - sccw

    def processGeometryIndexedTriangleFanSet(self, node):
        if False:
            for i in range(10):
                print('nop')
        fans = readIndex(node, 'index')
        ccw = int(self.startCoordMesh(node, sum([len(fan) - 2 for fan in fans])))
        for fan in fans:
            for i in range(1, len(fan) - 1):
                self.addTri(fan[0], fan[i + 1 - ccw], fan[i + ccw])

    def processGeometryTriangleSet(self, node):
        if False:
            while True:
                i = 10
        ccw = int(self.startCoordMesh(node, lambda num_vert: num_vert // 3))
        for i in range(0, self.getVertexCount(), 3):
            self.addTri(i + 1 - ccw, i + ccw, i + 2)

    def processGeometryTriangleStripSet(self, node):
        if False:
            print('Hello World!')
        strips = readIntArray(node, 'stripCount', [])
        ccw = int(self.startCoordMesh(node, sum([n - 2 for n in strips])))
        vb = 0
        for n in strips:
            sccw = ccw
            for i in range(n - 2):
                self.addTri(vb + i + 1 - sccw, vb + i + sccw, vb + i + 2)
                sccw = 1 - sccw
            vb += n

    def processGeometryTriangleFanSet(self, node):
        if False:
            i = 10
            return i + 15
        fans = readIntArray(node, 'fanCount', [])
        ccw = int(self.startCoordMesh(node, sum([n - 2 for n in fans])))
        vb = 0
        for n in fans:
            for i in range(1, n - 1):
                self.addTri(vb, vb + i + 1 - ccw, vb + i + ccw)
            vb += n

    def processGeometryQuadSet(self, node):
        if False:
            for i in range(10):
                print('nop')
        ccw = self.startCoordMesh(node, lambda num_vert: 2 * (num_vert // 4))
        for i in range(0, self.getVertexCount(), 4):
            self.addQuadFlip(i, i + 1, i + 2, i + 3, ccw)

    def processGeometryIndexedQuadSet(self, node):
        if False:
            for i in range(10):
                print('nop')
        index = readIntArray(node, 'index', [])
        num_quads = len(index) // 4
        ccw = self.startCoordMesh(node, num_quads * 2)
        for i in range(0, num_quads * 4, 4):
            self.addQuadFlip(index[i], index[i + 1], index[i + 2], index[i + 3], ccw)

    def processGeometryDisk2D(self, node):
        if False:
            print('Hello World!')
        innerRadius = readFloat(node, 'innerRadius', 0)
        outerRadius = readFloat(node, 'outerRadius', 1)
        n = readInt(node, 'subdivision', DEFAULT_SUBDIV)
        angle = 2 * pi / n
        self.reserveFaceAndVertexCount(n * 4 if innerRadius else n - 2, n * 2 if innerRadius else n)
        for i in range(n):
            s = sin(angle * i)
            c = cos(angle * i)
            self.addVertex(outerRadius * c, outerRadius * s, 0)
            if innerRadius:
                self.addVertex(innerRadius * c, innerRadius * s, 0)
                ni = (i + 1) % n
                self.addQuad(2 * i, 2 * ni, 2 * ni + 1, 2 * i + 1)
        if not innerRadius:
            for i in range(2, n):
                self.addTri(0, i - 1, i)

    def processGeometryRectangle2D(self, node):
        if False:
            i = 10
            return i + 15
        (x, y) = readFloatArray(node, 'size', (2, 2))
        self.reserveFaceAndVertexCount(2, 4)
        self.addVertex(-x / 2, -y / 2, 0)
        self.addVertex(x / 2, -y / 2, 0)
        self.addVertex(x / 2, y / 2, 0)
        self.addVertex(-x / 2, y / 2, 0)
        self.addQuad(0, 1, 2, 3)

    def processGeometryTriangleSet2D(self, node):
        if False:
            i = 10
            return i + 15
        verts = readFloatArray(node, 'vertices', ())
        num_faces = len(verts) // 6
        verts = [(verts[i], verts[i + 1], 0) for i in range(0, 6 * num_faces, 2)]
        self.reserveFaceAndVertexCount(num_faces, num_faces * 3)
        for vert in verts:
            self.addVertex(*vert)
        for i in range(0, num_faces * 3, 3):
            a = Vector(*verts[i + 2]) - Vector(*verts[i])
            b = Vector(*verts[i + 1]) - Vector(*verts[i])
            self.addTriFlip(i, i + 1, i + 2, a.x * b.y > a.y * b.x)

    def processGeometryIndexedFaceSet(self, node):
        if False:
            print('Hello World!')
        faces = readIndex(node, 'coordIndex')
        ccw = self.startCoordMesh(node, sum([len(face) - 2 for face in faces]))
        for face in faces:
            if len(face) == 3:
                self.addTriFlip(face[0], face[1], face[2], ccw)
            elif len(face) > 3:
                self.addFace(face, ccw)
    geometry_importers = {'IndexedFaceSet': processGeometryIndexedFaceSet, 'IndexedTriangleSet': processGeometryIndexedTriangleSet, 'IndexedTriangleStripSet': processGeometryIndexedTriangleStripSet, 'IndexedTriangleFanSet': processGeometryIndexedTriangleFanSet, 'TriangleSet': processGeometryTriangleSet, 'TriangleStripSet': processGeometryTriangleStripSet, 'TriangleFanSet': processGeometryTriangleFanSet, 'QuadSet': processGeometryQuadSet, 'IndexedQuadSet': processGeometryIndexedQuadSet, 'TriangleSet2D': processGeometryTriangleSet2D, 'Rectangle2D': processGeometryRectangle2D, 'Disk2D': processGeometryDisk2D, 'ElevationGrid': processGeometryElevationGrid, 'Extrusion': processGeometryExtrusion, 'Sphere': processGeometrySphere, 'Box': processGeometryBox, 'Cylinder': processGeometryCylinder, 'Cone': processGeometryCone}

    def readVertices(self, node):
        if False:
            for i in range(10):
                print('nop')
        for c in node:
            if c.tag == 'Coordinate':
                c = self.resolveDefUse(c)
                if c is not None:
                    pt = c.attrib.get('point')
                    if pt:
                        co = [float(x) for vec in pt.split(',') for x in vec.split()]
                        num_verts = len(co) // 3
                        self.verts = numpy.empty((4, num_verts), dtype=numpy.float32)
                        self.verts[3, :] = numpy.ones(num_verts, dtype=numpy.float32)
                        for i in range(num_verts):
                            self.verts[:3, i] = co[3 * i:3 * i + 3]

    def reserveFaceAndVertexCount(self, num_faces, num_verts):
        if False:
            while True:
                i = 10
        self.verts = numpy.zeros((4, num_verts), dtype=numpy.float32)
        self.verts[3, :] = numpy.ones(num_verts, dtype=numpy.float32)
        self.num_verts = 0
        self.reserveFaceCount(num_faces)

    def reserveFaceCount(self, num_faces):
        if False:
            for i in range(10):
                print('nop')
        self.faces = numpy.zeros((num_faces, 3), dtype=numpy.int32)
        self.num_faces = 0

    def getVertexCount(self):
        if False:
            print('Hello World!')
        return self.verts.shape[1]

    def addVertex(self, x, y, z):
        if False:
            while True:
                i = 10
        self.verts[0, self.num_verts] = x
        self.verts[1, self.num_verts] = y
        self.verts[2, self.num_verts] = z
        self.num_verts += 1

    def addTri(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        self.faces[self.num_faces, 0] = self.index_base + a
        self.faces[self.num_faces, 1] = self.index_base + b
        self.faces[self.num_faces, 2] = self.index_base + c
        self.num_faces += 1

    def addTriFlip(self, a, b, c, ccw):
        if False:
            return 10
        if ccw:
            self.addTri(a, b, c)
        else:
            self.addTri(b, a, c)

    def addQuad(self, a, b, c, d):
        if False:
            i = 10
            return i + 15
        self.addTri(a, b, c)
        self.addTri(c, d, a)

    def addQuadFlip(self, a, b, c, d, ccw):
        if False:
            return 10
        if ccw:
            self.addTri(a, b, c)
            self.addTri(c, d, a)
        else:
            self.addTri(a, c, b)
            self.addTri(c, a, d)

    def addFace(self, indices, ccw):
        if False:
            i = 10
            return i + 15
        face = [Vector(data=self.verts[0:3, i]) for i in indices]
        normal = findOuterNormal(face)
        if not normal:
            return
        n = len(face)
        vi = [i for i in range(n)]
        while n > 3:
            max_cos = EPSILON
            i_min = 0
            for i in range(n):
                inext = (i + 1) % n
                iprev = (i + n - 1) % n
                v = face[vi[i]]
                next = face[vi[inext]] - v
                prev = face[vi[iprev]] - v
                nextXprev = next.cross(prev)
                if nextXprev.dot(normal) > EPSILON:
                    cos = next.dot(prev) / (next.length() * prev.length())
                    if cos > max_cos:
                        no_points_inside = True
                        for j in range(n):
                            if j != i and j != iprev and (j != inext):
                                vx = face[vi[j]] - v
                                if pointInsideTriangle(vx, next, prev, nextXprev):
                                    no_points_inside = False
                                    break
                        if no_points_inside:
                            max_cos = cos
                            i_min = i
            self.addTriFlip(indices[vi[(i_min + n - 1) % n]], indices[vi[i_min]], indices[vi[(i_min + 1) % n]], ccw)
            vi.pop(i_min)
            n -= 1
        self.addTriFlip(indices[vi[0]], indices[vi[1]], indices[vi[2]], ccw)

def readFloatArray(node, attr, default):
    if False:
        return 10
    s = node.attrib.get(attr)
    if not s:
        return default
    return [float(x) for x in s.split()]

def readIntArray(node, attr, default):
    if False:
        i = 10
        return i + 15
    s = node.attrib.get(attr)
    if not s:
        return default
    return [int(x, 0) for x in s.split()]

def readFloat(node, attr, default):
    if False:
        i = 10
        return i + 15
    s = node.attrib.get(attr)
    if not s:
        return default
    return float(s)

def readInt(node, attr, default):
    if False:
        i = 10
        return i + 15
    s = node.attrib.get(attr)
    if not s:
        return default
    return int(s, 0)

def readBoolean(node, attr, default):
    if False:
        return 10
    s = node.attrib.get(attr)
    if not s:
        return default
    return s.lower() == 'true'

def readVector(node, attr, default):
    if False:
        return 10
    v = readFloatArray(node, attr, default)
    return Vector(v[0], v[1], v[2])

def readRotation(node, attr, default):
    if False:
        while True:
            i = 10
    v = readFloatArray(node, attr, default)
    return (v[3], Vector(v[0], v[1], v[2]))

def readIndex(node, attr):
    if False:
        i = 10
        return i + 15
    v = readIntArray(node, attr, [])
    chunks = []
    chunk = []
    for i in v:
        if i == -1:
            if chunk:
                chunks.append(chunk)
                chunk = []
        else:
            chunk.append(i)
    if chunk:
        chunks.append(chunk)
    return chunks

def findOuterNormal(face):
    if False:
        print('Hello World!')
    n = len(face)
    for i in range(n):
        for j in range(i + 1, n):
            edge = face[j] - face[i]
            if edge.length() > EPSILON:
                edge = edge.normalized()
                prev_rejection = Vector()
                is_outer = True
                for k in range(n):
                    if k != i and k != j:
                        pt = face[k] - face[i]
                        pte = pt.dot(edge)
                        rejection = pt - edge * pte
                        if rejection.dot(prev_rejection) < -EPSILON:
                            is_outer = False
                            break
                        elif rejection.length() > prev_rejection.length():
                            prev_rejection = rejection
                if is_outer:
                    return edge.cross(prev_rejection)
    return False

def ratio(a, b):
    if False:
        print('Hello World!')
    if b.x > EPSILON or b.x < -EPSILON:
        return a.x / b.x
    elif b.y > EPSILON or b.y < -EPSILON:
        return a.y / b.y
    else:
        return a.z / b.z

def pointInsideTriangle(vx, next, prev, nextXprev):
    if False:
        return 10
    vxXprev = vx.cross(prev)
    r = ratio(vxXprev, nextXprev)
    if r < 0:
        return False
    vxXnext = vx.cross(next)
    s = -ratio(vxXnext, nextXprev)
    return s > 0 and s + r < 1