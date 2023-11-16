import math
import sys
import getopt
TOLERANCE = 1e-09
BIG_FLOAT = 1e+38
if sys.version > '3':
    PY3 = True
else:
    PY3 = False

class Context(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.doPrint = 0
        self.debug = 0
        self.extent = ()
        self.triangulate = False
        self.vertices = []
        self.lines = []
        self.edges = []
        self.triangles = []
        self.polygons = {}

    def getClipEdges(self):
        if False:
            i = 10
            return i + 15
        (xmin, xmax, ymin, ymax) = self.extent
        clipEdges = []
        for edge in self.edges:
            equation = self.lines[edge[0]]
            if edge[1] != -1 and edge[2] != -1:
                (x1, y1) = (self.vertices[edge[1]][0], self.vertices[edge[1]][1])
                (x2, y2) = (self.vertices[edge[2]][0], self.vertices[edge[2]][1])
                (pt1, pt2) = ((x1, y1), (x2, y2))
                (inExtentP1, inExtentP2) = (self.inExtent(x1, y1), self.inExtent(x2, y2))
                if inExtentP1 and inExtentP2:
                    clipEdges.append((pt1, pt2))
                elif inExtentP1 and (not inExtentP2):
                    pt2 = self.clipLine(x1, y1, equation, leftDir=False)
                    clipEdges.append((pt1, pt2))
                elif not inExtentP1 and inExtentP2:
                    pt1 = self.clipLine(x2, y2, equation, leftDir=True)
                    clipEdges.append((pt1, pt2))
            else:
                if edge[1] != -1:
                    (x1, y1) = (self.vertices[edge[1]][0], self.vertices[edge[1]][1])
                    leftDir = False
                else:
                    (x1, y1) = (self.vertices[edge[2]][0], self.vertices[edge[2]][1])
                    leftDir = True
                if self.inExtent(x1, y1):
                    pt1 = (x1, y1)
                    pt2 = self.clipLine(x1, y1, equation, leftDir)
                    clipEdges.append((pt1, pt2))
        return clipEdges

    def getClipPolygons(self, closePoly):
        if False:
            while True:
                i = 10
        (xmin, xmax, ymin, ymax) = self.extent
        poly = {}
        for (inPtsIdx, edges) in self.polygons.items():
            clipEdges = []
            for edge in edges:
                equation = self.lines[edge[0]]
                if edge[1] != -1 and edge[2] != -1:
                    (x1, y1) = (self.vertices[edge[1]][0], self.vertices[edge[1]][1])
                    (x2, y2) = (self.vertices[edge[2]][0], self.vertices[edge[2]][1])
                    (pt1, pt2) = ((x1, y1), (x2, y2))
                    (inExtentP1, inExtentP2) = (self.inExtent(x1, y1), self.inExtent(x2, y2))
                    if inExtentP1 and inExtentP2:
                        clipEdges.append((pt1, pt2))
                    elif inExtentP1 and (not inExtentP2):
                        pt2 = self.clipLine(x1, y1, equation, leftDir=False)
                        clipEdges.append((pt1, pt2))
                    elif not inExtentP1 and inExtentP2:
                        pt1 = self.clipLine(x2, y2, equation, leftDir=True)
                        clipEdges.append((pt1, pt2))
                else:
                    if edge[1] != -1:
                        (x1, y1) = (self.vertices[edge[1]][0], self.vertices[edge[1]][1])
                        leftDir = False
                    else:
                        (x1, y1) = (self.vertices[edge[2]][0], self.vertices[edge[2]][1])
                        leftDir = True
                    if self.inExtent(x1, y1):
                        pt1 = (x1, y1)
                        pt2 = self.clipLine(x1, y1, equation, leftDir)
                        clipEdges.append((pt1, pt2))
            (polyPts, complete) = self.orderPts(clipEdges)
            if not complete:
                startPt = polyPts[0]
                endPt = polyPts[-1]
                if startPt[0] == endPt[0] or startPt[1] == endPt[1]:
                    polyPts.append(polyPts[0])
                else:
                    if startPt[0] == xmin and endPt[1] == ymax or (endPt[0] == xmin and startPt[1] == ymax):
                        polyPts.append((xmin, ymax))
                        polyPts.append(polyPts[0])
                    if startPt[0] == xmax and endPt[1] == ymax or (endPt[0] == xmax and startPt[1] == ymax):
                        polyPts.append((xmax, ymax))
                        polyPts.append(polyPts[0])
                    if startPt[0] == xmax and endPt[1] == ymin or (endPt[0] == xmax and startPt[1] == ymin):
                        polyPts.append((xmax, ymin))
                        polyPts.append(polyPts[0])
                    if startPt[0] == xmin and endPt[1] == ymin or (endPt[0] == xmin and startPt[1] == ymin):
                        polyPts.append((xmin, ymin))
                        polyPts.append(polyPts[0])
            if not closePoly:
                polyPts = polyPts[:-1]
            poly[inPtsIdx] = polyPts
        return poly

    def clipLine(self, x1, y1, equation, leftDir):
        if False:
            return 10
        (xmin, xmax, ymin, ymax) = self.extent
        (a, b, c) = equation
        if b == 0:
            if leftDir:
                return (x1, ymax)
            else:
                return (x1, ymin)
        elif a == 0:
            if leftDir:
                return (xmin, y1)
            else:
                return (xmax, y1)
        else:
            y2_at_xmin = (c - a * xmin) / b
            y2_at_xmax = (c - a * xmax) / b
            x2_at_ymin = (c - b * ymin) / a
            x2_at_ymax = (c - b * ymax) / a
            intersectPts = []
            if ymin <= y2_at_xmin <= ymax:
                intersectPts.append((xmin, y2_at_xmin))
            if ymin <= y2_at_xmax <= ymax:
                intersectPts.append((xmax, y2_at_xmax))
            if xmin <= x2_at_ymin <= xmax:
                intersectPts.append((x2_at_ymin, ymin))
            if xmin <= x2_at_ymax <= xmax:
                intersectPts.append((x2_at_ymax, ymax))
            intersectPts = set(intersectPts)
            if leftDir:
                pt = min(intersectPts)
            else:
                pt = max(intersectPts)
            return pt

    def inExtent(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        (xmin, xmax, ymin, ymax) = self.extent
        return x >= xmin and x <= xmax and (y >= ymin) and (y <= ymax)

    def orderPts(self, edges):
        if False:
            print('Hello World!')
        poly = []
        pts = []
        for edge in edges:
            pts.extend([pt for pt in edge])
        try:
            (startPt, endPt) = [pt for pt in pts if pts.count(pt) < 2]
        except:
            complete = True
            firstIdx = 0
            poly.append(edges[0][0])
            poly.append(edges[0][1])
        else:
            complete = False
            for (i, edge) in enumerate(edges):
                if startPt in edge:
                    firstIdx = i
                    break
            poly.append(edges[firstIdx][0])
            poly.append(edges[firstIdx][1])
            if poly[0] != startPt:
                poly.reverse()
        del edges[firstIdx]
        while edges:
            currentPt = poly[-1]
            for (i, edge) in enumerate(edges):
                if currentPt == edge[0]:
                    poly.append(edge[1])
                    break
                elif currentPt == edge[1]:
                    poly.append(edge[0])
                    break
            del edges[i]
        return (poly, complete)

    def setClipBuffer(self, xpourcent, ypourcent):
        if False:
            i = 10
            return i + 15
        (xmin, xmax, ymin, ymax) = self.extent
        witdh = xmax - xmin
        height = ymax - ymin
        xmin = xmin - witdh * xpourcent / 100
        xmax = xmax + witdh * xpourcent / 100
        ymin = ymin - height * ypourcent / 100
        ymax = ymax + height * ypourcent / 100
        self.extent = (xmin, xmax, ymin, ymax)

    def outSite(self, s):
        if False:
            for i in range(10):
                print('nop')
        if self.debug:
            print('site (%d) at %f %f' % (s.sitenum, s.x, s.y))
        elif self.triangulate:
            pass
        elif self.doPrint:
            print('s %f %f' % (s.x, s.y))

    def outVertex(self, s):
        if False:
            print('Hello World!')
        self.vertices.append((s.x, s.y))
        if self.debug:
            print('vertex(%d) at %f %f' % (s.sitenum, s.x, s.y))
        elif self.triangulate:
            pass
        elif self.doPrint:
            print('v %f %f' % (s.x, s.y))

    def outTriple(self, s1, s2, s3):
        if False:
            print('Hello World!')
        self.triangles.append((s1.sitenum, s2.sitenum, s3.sitenum))
        if self.debug:
            print('circle through left=%d right=%d bottom=%d' % (s1.sitenum, s2.sitenum, s3.sitenum))
        elif self.triangulate and self.doPrint:
            print('%d %d %d' % (s1.sitenum, s2.sitenum, s3.sitenum))

    def outBisector(self, edge):
        if False:
            print('Hello World!')
        self.lines.append((edge.a, edge.b, edge.c))
        if self.debug:
            print('line(%d) %gx+%gy=%g, bisecting %d %d' % (edge.edgenum, edge.a, edge.b, edge.c, edge.reg[0].sitenum, edge.reg[1].sitenum))
        elif self.doPrint:
            print('l %f %f %f' % (edge.a, edge.b, edge.c))

    def outEdge(self, edge):
        if False:
            print('Hello World!')
        sitenumL = -1
        if edge.ep[Edge.LE] is not None:
            sitenumL = edge.ep[Edge.LE].sitenum
        sitenumR = -1
        if edge.ep[Edge.RE] is not None:
            sitenumR = edge.ep[Edge.RE].sitenum
        if edge.reg[0].sitenum not in self.polygons:
            self.polygons[edge.reg[0].sitenum] = []
        if edge.reg[1].sitenum not in self.polygons:
            self.polygons[edge.reg[1].sitenum] = []
        self.polygons[edge.reg[0].sitenum].append((edge.edgenum, sitenumL, sitenumR))
        self.polygons[edge.reg[1].sitenum].append((edge.edgenum, sitenumL, sitenumR))
        self.edges.append((edge.edgenum, sitenumL, sitenumR))
        if not self.triangulate:
            if self.doPrint:
                print('e %d' % edge.edgenum)
                print(' %d ' % sitenumL)
                print('%d' % sitenumR)

def voronoi(siteList, context):
    if False:
        i = 10
        return i + 15
    context.extent = siteList.extent
    edgeList = EdgeList(siteList.xmin, siteList.xmax, len(siteList))
    priorityQ = PriorityQueue(siteList.ymin, siteList.ymax, len(siteList))
    siteIter = siteList.iterator()
    bottomsite = siteIter.next()
    context.outSite(bottomsite)
    newsite = siteIter.next()
    minpt = Site(-BIG_FLOAT, -BIG_FLOAT)
    while True:
        if not priorityQ.isEmpty():
            minpt = priorityQ.getMinPt()
        if newsite and (priorityQ.isEmpty() or newsite < minpt):
            context.outSite(newsite)
            lbnd = edgeList.leftbnd(newsite)
            rbnd = lbnd.right
            bot = lbnd.rightreg(bottomsite)
            edge = Edge.bisect(bot, newsite)
            context.outBisector(edge)
            bisector = Halfedge(edge, Edge.LE)
            edgeList.insert(lbnd, bisector)
            p = lbnd.intersect(bisector)
            if p is not None:
                priorityQ.delete(lbnd)
                priorityQ.insert(lbnd, p, newsite.distance(p))
            lbnd = bisector
            bisector = Halfedge(edge, Edge.RE)
            edgeList.insert(lbnd, bisector)
            p = bisector.intersect(rbnd)
            if p is not None:
                priorityQ.insert(bisector, p, newsite.distance(p))
            newsite = siteIter.next()
        elif not priorityQ.isEmpty():
            lbnd = priorityQ.popMinHalfedge()
            llbnd = lbnd.left
            rbnd = lbnd.right
            rrbnd = rbnd.right
            bot = lbnd.leftreg(bottomsite)
            top = rbnd.rightreg(bottomsite)
            mid = lbnd.rightreg(bottomsite)
            context.outTriple(bot, top, mid)
            v = lbnd.vertex
            siteList.setSiteNumber(v)
            context.outVertex(v)
            if lbnd.edge.setEndpoint(lbnd.pm, v):
                context.outEdge(lbnd.edge)
            if rbnd.edge.setEndpoint(rbnd.pm, v):
                context.outEdge(rbnd.edge)
            edgeList.delete(lbnd)
            priorityQ.delete(rbnd)
            edgeList.delete(rbnd)
            pm = Edge.LE
            if bot.y > top.y:
                (bot, top) = (top, bot)
                pm = Edge.RE
            edge = Edge.bisect(bot, top)
            context.outBisector(edge)
            bisector = Halfedge(edge, pm)
            edgeList.insert(llbnd, bisector)
            if edge.setEndpoint(Edge.RE - pm, v):
                context.outEdge(edge)
            p = llbnd.intersect(bisector)
            if p is not None:
                priorityQ.delete(llbnd)
                priorityQ.insert(llbnd, p, bot.distance(p))
            p = bisector.intersect(rrbnd)
            if p is not None:
                priorityQ.insert(bisector, p, bot.distance(p))
        else:
            break
    he = edgeList.leftend.right
    while he is not edgeList.rightend:
        context.outEdge(he.edge)
        he = he.right
    Edge.EDGE_NUM = 0

def isEqual(a, b, relativeError=TOLERANCE):
    if False:
        i = 10
        return i + 15
    norm = max(abs(a), abs(b))
    return norm < relativeError or abs(a - b) < relativeError * norm

class Site(object):

    def __init__(self, x=0.0, y=0.0, sitenum=0):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y
        self.sitenum = sitenum

    def dump(self):
        if False:
            return 10
        print('Site #%d (%g, %g)' % (self.sitenum, self.x, self.y))

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.y < other.y:
            return True
        elif self.y > other.y:
            return False
        elif self.x < other.x:
            return True
        elif self.x > other.x:
            return False
        else:
            return False

    def __eq__(self, other):
        if False:
            return 10
        if self.y == other.y and self.x == other.x:
            return True

    def distance(self, other):
        if False:
            i = 10
            return i + 15
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

class Edge(object):
    LE = 0
    RE = 1
    EDGE_NUM = 0
    DELETED = {}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.ep = [None, None]
        self.reg = [None, None]
        self.edgenum = 0

    def dump(self):
        if False:
            while True:
                i = 10
        print('(#%d a=%g, b=%g, c=%g)' % (self.edgenum, self.a, self.b, self.c))
        print('ep', self.ep)
        print('reg', self.reg)

    def setEndpoint(self, lrFlag, site):
        if False:
            i = 10
            return i + 15
        self.ep[lrFlag] = site
        if self.ep[Edge.RE - lrFlag] is None:
            return False
        return True

    @staticmethod
    def bisect(s1, s2):
        if False:
            print('Hello World!')
        newedge = Edge()
        newedge.reg[0] = s1
        newedge.reg[1] = s2
        dx = float(s2.x - s1.x)
        dy = float(s2.y - s1.y)
        adx = abs(dx)
        ady = abs(dy)
        newedge.c = float(s1.x * dx + s1.y * dy + (dx * dx + dy * dy) * 0.5)
        if adx > ady:
            newedge.a = 1.0
            newedge.b = dy / dx
            newedge.c /= dx
        else:
            newedge.b = 1.0
            newedge.a = dx / dy
            newedge.c /= dy
        newedge.edgenum = Edge.EDGE_NUM
        Edge.EDGE_NUM += 1
        return newedge

class Halfedge(object):

    def __init__(self, edge=None, pm=Edge.LE):
        if False:
            return 10
        self.left = None
        self.right = None
        self.qnext = None
        self.edge = edge
        self.pm = pm
        self.vertex = None
        self.ystar = BIG_FLOAT

    def dump(self):
        if False:
            while True:
                i = 10
        print('Halfedge--------------------------')
        print('left: ', self.left)
        print('right: ', self.right)
        print('edge: ', self.edge)
        print('pm: ', self.pm)
        (print('vertex: '),)
        if self.vertex:
            self.vertex.dump()
        else:
            print('None')
        print('ystar: ', self.ystar)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if self.ystar < other.ystar:
            return True
        elif self.ystar > other.ystar:
            return False
        elif self.vertex.x < other.vertex.x:
            return True
        elif self.vertex.x > other.vertex.x:
            return False
        else:
            return False

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self.ystar == other.ystar and self.vertex.x == other.vertex.x:
            return True

    def leftreg(self, default):
        if False:
            return 10
        if not self.edge:
            return default
        elif self.pm == Edge.LE:
            return self.edge.reg[Edge.LE]
        else:
            return self.edge.reg[Edge.RE]

    def rightreg(self, default):
        if False:
            return 10
        if not self.edge:
            return default
        elif self.pm == Edge.LE:
            return self.edge.reg[Edge.RE]
        else:
            return self.edge.reg[Edge.LE]

    def isPointRightOf(self, pt):
        if False:
            for i in range(10):
                print('nop')
        e = self.edge
        topsite = e.reg[1]
        right_of_site = pt.x > topsite.x
        if right_of_site and self.pm == Edge.LE:
            return True
        if not right_of_site and self.pm == Edge.RE:
            return False
        if e.a == 1.0:
            dyp = pt.y - topsite.y
            dxp = pt.x - topsite.x
            fast = 0
            if not right_of_site and e.b < 0.0 or (right_of_site and e.b >= 0.0):
                above = dyp >= e.b * dxp
                fast = above
            else:
                above = pt.x + pt.y * e.b > e.c
                if e.b < 0.0:
                    above = not above
                if not above:
                    fast = 1
            if not fast:
                dxs = topsite.x - e.reg[0].x
                above = e.b * (dxp * dxp - dyp * dyp) < dxs * dyp * (1.0 + 2.0 * dxp / dxs + e.b * e.b)
                if e.b < 0.0:
                    above = not above
        else:
            yl = e.c - e.a * pt.x
            t1 = pt.y - yl
            t2 = pt.x - topsite.x
            t3 = yl - topsite.y
            above = t1 * t1 > t2 * t2 + t3 * t3
        if self.pm == Edge.LE:
            return above
        else:
            return not above

    def intersect(self, other):
        if False:
            while True:
                i = 10
        e1 = self.edge
        e2 = other.edge
        if e1 is None or e2 is None:
            return None
        if e1.reg[1] is e2.reg[1]:
            return None
        d = e1.a * e2.b - e1.b * e2.a
        if isEqual(d, 0.0):
            return None
        xint = (e1.c * e2.b - e2.c * e1.b) / d
        yint = (e2.c * e1.a - e1.c * e2.a) / d
        if e1.reg[1] < e2.reg[1]:
            he = self
            e = e1
        else:
            he = other
            e = e2
        rightOfSite = xint >= e.reg[1].x
        if rightOfSite and he.pm == Edge.LE or (not rightOfSite and he.pm == Edge.RE):
            return None
        return Site(xint, yint)

class EdgeList(object):

    def __init__(self, xmin, xmax, nsites):
        if False:
            while True:
                i = 10
        if xmin > xmax:
            (xmin, xmax) = (xmax, xmin)
        self.hashsize = int(2 * math.sqrt(nsites + 4))
        self.xmin = xmin
        self.deltax = float(xmax - xmin)
        self.hash = [None] * self.hashsize
        self.leftend = Halfedge()
        self.rightend = Halfedge()
        self.leftend.right = self.rightend
        self.rightend.left = self.leftend
        self.hash[0] = self.leftend
        self.hash[-1] = self.rightend

    def insert(self, left, he):
        if False:
            for i in range(10):
                print('nop')
        he.left = left
        he.right = left.right
        left.right.left = he
        left.right = he

    def delete(self, he):
        if False:
            return 10
        he.left.right = he.right
        he.right.left = he.left
        he.edge = Edge.DELETED

    def gethash(self, b):
        if False:
            return 10
        if b < 0 or b >= self.hashsize:
            return None
        he = self.hash[b]
        if he is None or he.edge is not Edge.DELETED:
            return he
        self.hash[b] = None
        return None

    def leftbnd(self, pt):
        if False:
            i = 10
            return i + 15
        bucket = int((pt.x - self.xmin) / self.deltax * self.hashsize)
        if bucket < 0:
            bucket = 0
        if bucket >= self.hashsize:
            bucket = self.hashsize - 1
        he = self.gethash(bucket)
        if he is None:
            i = 1
            while True:
                he = self.gethash(bucket - i)
                if he is not None:
                    break
                he = self.gethash(bucket + i)
                if he is not None:
                    break
                i += 1
        if he is self.leftend or (he is not self.rightend and he.isPointRightOf(pt)):
            he = he.right
            while he is not self.rightend and he.isPointRightOf(pt):
                he = he.right
            he = he.left
        else:
            he = he.left
            while he is not self.leftend and (not he.isPointRightOf(pt)):
                he = he.left
        if bucket > 0 and bucket < self.hashsize - 1:
            self.hash[bucket] = he
        return he

class PriorityQueue(object):

    def __init__(self, ymin, ymax, nsites):
        if False:
            print('Hello World!')
        self.ymin = ymin
        self.deltay = ymax - ymin
        self.hashsize = int(4 * math.sqrt(nsites))
        self.count = 0
        self.minidx = 0
        self.hash = []
        for i in range(self.hashsize):
            self.hash.append(Halfedge())

    def __len__(self):
        if False:
            return 10
        return self.count

    def isEmpty(self):
        if False:
            i = 10
            return i + 15
        return self.count == 0

    def insert(self, he, site, offset):
        if False:
            print('Hello World!')
        he.vertex = site
        he.ystar = site.y + offset
        last = self.hash[self.getBucket(he)]
        next = last.qnext
        while next is not None and he > next:
            last = next
            next = last.qnext
        he.qnext = last.qnext
        last.qnext = he
        self.count += 1

    def delete(self, he):
        if False:
            i = 10
            return i + 15
        if he.vertex is not None:
            last = self.hash[self.getBucket(he)]
            while last.qnext is not he:
                last = last.qnext
            last.qnext = he.qnext
            self.count -= 1
            he.vertex = None

    def getBucket(self, he):
        if False:
            while True:
                i = 10
        bucket = int((he.ystar - self.ymin) / self.deltay * self.hashsize)
        if bucket < 0:
            bucket = 0
        if bucket >= self.hashsize:
            bucket = self.hashsize - 1
        if bucket < self.minidx:
            self.minidx = bucket
        return bucket

    def getMinPt(self):
        if False:
            for i in range(10):
                print('nop')
        while self.hash[self.minidx].qnext is None:
            self.minidx += 1
        he = self.hash[self.minidx].qnext
        x = he.vertex.x
        y = he.ystar
        return Site(x, y)

    def popMinHalfedge(self):
        if False:
            i = 10
            return i + 15
        curr = self.hash[self.minidx].qnext
        self.hash[self.minidx].qnext = curr.qnext
        self.count -= 1
        return curr

class SiteList(object):

    def __init__(self, pointList):
        if False:
            while True:
                i = 10
        self.__sites = []
        self.__sitenum = 0
        self.__xmin = min([pt.x for pt in pointList])
        self.__ymin = min([pt.y for pt in pointList])
        self.__xmax = max([pt.x for pt in pointList])
        self.__ymax = max([pt.y for pt in pointList])
        self.__extent = (self.__xmin, self.__xmax, self.__ymin, self.__ymax)
        for (i, pt) in enumerate(pointList):
            self.__sites.append(Site(pt.x, pt.y, i))
        self.__sites.sort()

    def setSiteNumber(self, site):
        if False:
            for i in range(10):
                print('nop')
        site.sitenum = self.__sitenum
        self.__sitenum += 1

    class Iterator(object):

        def __init__(this, lst):
            if False:
                i = 10
                return i + 15
            this.generator = (s for s in lst)

        def __iter__(this):
            if False:
                while True:
                    i = 10
            return this

        def next(this):
            if False:
                i = 10
                return i + 15
            try:
                if PY3:
                    return this.generator.__next__()
                else:
                    return this.generator.next()
            except StopIteration:
                return None

    def iterator(self):
        if False:
            while True:
                i = 10
        return SiteList.Iterator(self.__sites)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return SiteList.Iterator(self.__sites)

    def __len__(self):
        if False:
            return 10
        return len(self.__sites)

    def _getxmin(self):
        if False:
            print('Hello World!')
        return self.__xmin

    def _getymin(self):
        if False:
            return 10
        return self.__ymin

    def _getxmax(self):
        if False:
            return 10
        return self.__xmax

    def _getymax(self):
        if False:
            while True:
                i = 10
        return self.__ymax

    def _getextent(self):
        if False:
            return 10
        return self.__extent
    xmin = property(_getxmin)
    ymin = property(_getymin)
    xmax = property(_getxmax)
    ymax = property(_getymax)
    extent = property(_getextent)

def computeVoronoiDiagram(points, xBuff=0, yBuff=0, polygonsOutput=False, formatOutput=False, closePoly=True):
    if False:
        while True:
            i = 10
    '\n\tTakes :\n\t\t- a list of point objects (which must have x and y fields).\n\t\t- x and y buffer values which are the expansion percentages of the bounding box rectangle including all input points.\n\t\tReturns :\n\t\t- With default options : \n\t\t  A list of 2-tuples, representing the two points of each Voronoi diagram edge.\n\t\t  Each point contains 2-tuples which are the x,y coordinates of point.\n\t\t  if formatOutput is True, returns : \n\t\t\t\t- a list of 2-tuples, which are the x,y coordinates of the Voronoi diagram vertices.\n\t\t\t\t- and a list of 2-tuples (v1, v2) representing edges of the Voronoi diagram.\n\t\t\t\t  v1 and v2 are the indices of the vertices at the end of the edge.\n\t\t- If polygonsOutput option is True, returns :\n\t\t  A dictionary of polygons, keys are the indices of the input points,\n\t\t  values contains n-tuples representing the n points of each Voronoi diagram polygon.\n\t\t  Each point contains 2-tuples which are the x,y coordinates of point.\n\t\t  if formatOutput is True, returns : \n\t\t\t\t- A list of 2-tuples, which are the x,y coordinates of the Voronoi diagram vertices.\n\t\t\t\t- and a dictionary of input points indices. Values contains n-tuples representing the n points of each Voronoi diagram polygon.\n\t\t\t\t  Each tuple contains the vertex indices of the polygon vertices.\n\t\t- if closePoly is True then, in the list of points of a polygon, last point will be the same of first point\n\t'
    siteList = SiteList(points)
    context = Context()
    voronoi(siteList, context)
    context.setClipBuffer(xBuff, yBuff)
    if not polygonsOutput:
        clipEdges = context.getClipEdges()
        if formatOutput:
            (vertices, edgesIdx) = formatEdgesOutput(clipEdges)
            return (vertices, edgesIdx)
        else:
            return clipEdges
    else:
        clipPolygons = context.getClipPolygons(closePoly)
        if formatOutput:
            (vertices, polyIdx) = formatPolygonsOutput(clipPolygons)
            return (vertices, polyIdx)
        else:
            return clipPolygons

def formatEdgesOutput(edges):
    if False:
        print('Hello World!')
    pts = []
    for edge in edges:
        pts.extend(edge)
    pts = set(pts)
    valuesIdxDict = dict(zip(pts, range(len(pts))))
    edgesIdx = []
    for edge in edges:
        edgesIdx.append([valuesIdxDict[pt] for pt in edge])
    return (list(pts), edgesIdx)

def formatPolygonsOutput(polygons):
    if False:
        i = 10
        return i + 15
    pts = []
    for poly in polygons.values():
        pts.extend(poly)
    pts = set(pts)
    valuesIdxDict = dict(zip(pts, range(len(pts))))
    polygonsIdx = {}
    for (inPtsIdx, poly) in polygons.items():
        polygonsIdx[inPtsIdx] = [valuesIdxDict[pt] for pt in poly]
    return (list(pts), polygonsIdx)

def computeDelaunayTriangulation(points):
    if False:
        for i in range(10):
            print('nop')
    ' Takes a list of point objects (which must have x and y fields).\n\t\tReturns a list of 3-tuples: the indices of the points that form a\n\t\tDelaunay triangle.\n\t'
    siteList = SiteList(points)
    context = Context()
    context.triangulate = True
    voronoi(siteList, context)
    return context.triangles