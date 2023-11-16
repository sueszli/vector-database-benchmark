from panda3d.core import ConfigVariableBool, NodePath, NurbsCurveEvaluator, Point3, RopeNode, VBase3, VBase4

class Rope(NodePath):
    """
    This class defines a NURBS curve whose control vertices are
    defined based on points relative to one or more nodes in space, so
    that the "rope" will animate as the nodes move around.  It uses
    the C++ RopeNode class to achieve fancy rendering effects like
    thick lines built from triangle strips.
    """
    showRope = ConfigVariableBool('show-rope', True, 'Set this to false to deactivate the display of ropes.')

    def __init__(self, name='Rope'):
        if False:
            print('Hello World!')
        self.ropeNode = RopeNode(name)
        self.curve = NurbsCurveEvaluator()
        self.ropeNode.setCurve(self.curve)
        NodePath.__init__(self, self.ropeNode)
        self.name = name
        self.order = 0
        self.verts = []
        self.knots = None

    def setup(self, order, verts, knots=None):
        if False:
            while True:
                i = 10
        "This must be called to define the shape of the curve\n        initially, and may be called again as needed to adjust the\n        curve's properties.\n\n        order must be either 1, 2, 3, or 4, and is one more than the\n        degree of the curve; most NURBS curves are order 4.\n\n        verts is a list of (NodePath, point) tuples, defining the\n        control vertices of the curve.  For each control vertex, the\n        NodePath may refer to an arbitrary node in the scene graph,\n        indicating the point should be interpreted in the coordinate\n        space of that node (and it will automatically move when the\n        node is moved), or it may be the empty NodePath or None to\n        indicate the point should be interpreted in the coordinate\n        space of the Rope itself.  Each point value may be either a\n        3-tuple or a 4-tuple (or a VBase3 or VBase4).  If it is a\n        3-component vector, it represents a 3-d point in space; a\n        4-component vector represents a point in 4-d homogeneous\n        space; that is to say, a 3-d point and an additional weight\n        factor (which should have been multiplied into the x y z\n        components).\n\n        verts may be a list of dictionaries instead of a list of\n        tuples.  In this case, each vertex dictionary may have any of\n        the following elements:\n\n          'node' : the NodePath indicating the coordinate space\n          'point' : the 3-D point relative to the node; default (0, 0, 0)\n          'color' : the color of the vertex, default (1, 1, 1, 1)\n          'thickness' : the thickness at the vertex, default 1\n\n        In order to enable the per-vertex color or thickness, you must\n        call rope.ropeNode.setUseVertexColor(1) or\n        rope.ropeNode.setUseVertexThickness(1).\n\n        knots is optional.  If specified, it should be a list of\n        floats, and should be of length len(verts) + order.  If it\n        is omitted, a default knot string is generated that consists\n        of the first (order - 1) and last (order - 1) values the\n        same, and the intermediate values incrementing by 1.\n        "
        self.order = order
        self.verts = verts
        self.knots = knots
        self.recompute()

    def recompute(self):
        if False:
            for i in range(10):
                print('nop')
        'Recomputes the curve after its properties have changed.\n        Normally it is not necessary for the user to call this\n        directly.'
        if not self.showRope:
            return
        numVerts = len(self.verts)
        self.curve.reset(numVerts)
        self.curve.setOrder(self.order)
        defaultNodePath = None
        defaultPoint = (0, 0, 0)
        defaultColor = (1, 1, 1, 1)
        defaultThickness = 1
        useVertexColor = self.ropeNode.getUseVertexColor()
        useVertexThickness = self.ropeNode.getUseVertexThickness()
        vcd = self.ropeNode.getVertexColorDimension()
        vtd = self.ropeNode.getVertexThicknessDimension()
        for i in range(numVerts):
            v = self.verts[i]
            if isinstance(v, tuple):
                (nodePath, point) = v
                color = defaultColor
                thickness = defaultThickness
            else:
                nodePath = v.get('node', defaultNodePath)
                point = v.get('point', defaultPoint)
                color = v.get('color', defaultColor)
                thickness = v.get('thickness', defaultThickness)
            if isinstance(point, tuple):
                if len(point) >= 4:
                    self.curve.setVertex(i, VBase4(point[0], point[1], point[2], point[3]))
                else:
                    self.curve.setVertex(i, VBase3(point[0], point[1], point[2]))
            else:
                self.curve.setVertex(i, point)
            if nodePath:
                self.curve.setVertexSpace(i, nodePath)
            if useVertexColor:
                self.curve.setExtendedVertex(i, vcd + 0, color[0])
                self.curve.setExtendedVertex(i, vcd + 1, color[1])
                self.curve.setExtendedVertex(i, vcd + 2, color[2])
                self.curve.setExtendedVertex(i, vcd + 3, color[3])
            if useVertexThickness:
                self.curve.setExtendedVertex(i, vtd, thickness)
        if self.knots is not None:
            for i in range(len(self.knots)):
                self.curve.setKnot(i, self.knots[i])
        self.ropeNode.resetBound(self)

    def getPoints(self, len):
        if False:
            print('Hello World!')
        'Returns a list of len points, evenly distributed in\n        parametric space on the rope, in the coordinate space of the\n        Rope itself.'
        result = self.curve.evaluate(self)
        startT = result.getStartT()
        sizeT = result.getEndT() - startT
        numPts = len
        ropePts = []
        for i in range(numPts):
            pt = Point3()
            result.evalPoint(sizeT * i / float(numPts - 1) + startT, pt)
            ropePts.append(pt)
        return ropePts