import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui

class GlassDB:
    """
    Database of dispersion coefficients for Schott glasses
     + Corning 7980
    """

    def __init__(self, fileName='schott_glasses.csv'):
        if False:
            while True:
                i = 10
        path = os.path.dirname(__file__)
        fh = gzip.open(os.path.join(path, 'schott_glasses.csv.gz'), 'rb')
        r = csv.reader(map(str, fh.readlines()))
        lines = [x for x in r]
        self.data = {}
        header = lines[0]
        for l in lines[1:]:
            info = {}
            for i in range(1, len(l)):
                info[header[i]] = l[i]
            self.data[l[0]] = info
        self.data['Corning7980'] = {'B1': 0.683740494, 'B2': 0.420323613, 'B3': 0.58502748, 'C1': 0.00460352869, 'C2': 0.0133968856, 'C3': 64.4932732, 'TAUI25/250': 0.95, 'TAUI25/1400': 0.98}
        for k in self.data:
            self.data[k]['ior_cache'] = {}

    def ior(self, glass, wl):
        if False:
            return 10
        '\n        Return the index of refraction for *glass* at wavelength *wl*.\n        \n        The *glass* argument must be a key in self.data.\n        '
        info = self.data[glass]
        cache = info['ior_cache']
        if wl not in cache:
            B = list(map(float, [info['B1'], info['B2'], info['B3']]))
            C = list(map(float, [info['C1'], info['C2'], info['C3']]))
            w2 = (wl / 1000.0) ** 2
            n = sqrt(1.0 + B[0] * w2 / (w2 - C[0]) + B[1] * w2 / (w2 - C[1]) + B[2] * w2 / (w2 - C[2]))
            cache[wl] = n
        return cache[wl]

    def transmissionCurve(self, glass):
        if False:
            return 10
        data = self.data[glass]
        keys = [int(x[7:]) for x in data.keys() if 'TAUI25' in x]
        keys.sort()
        curve = np.empty((2, len(keys)))
        for i in range(len(keys)):
            curve[0][i] = keys[i]
            key = 'TAUI25/%d' % keys[i]
            val = data[key]
            if val == '':
                val = 0
            else:
                val = float(val)
            curve[1][i] = val
        return curve
GLASSDB = GlassDB()

def wlPen(wl):
    if False:
        i = 10
        return i + 15
    'Return a pen representing the given wavelength'
    l1 = 400
    l2 = 700
    hue = np.clip((l2 - l1 - (wl - l1)) * 0.8 / (l2 - l1), 0, 0.8)
    val = 1.0
    if wl > 700:
        val = 1.0 * ((700 - wl) / 700.0 + 1)
    elif wl < 400:
        val = wl * 1.0 / 400.0
    color = pg.hsvColor(hue, 1.0, val)
    pen = pg.mkPen(color)
    return pen

class ParamObj(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__params = {}

    def __setitem__(self, item, val):
        if False:
            print('Hello World!')
        self.setParam(item, val)

    def setParam(self, param, val):
        if False:
            print('Hello World!')
        self.setParams(**{param: val})

    def setParams(self, **params):
        if False:
            print('Hello World!')
        'Set parameters for this optic. This is a good function to override for subclasses.'
        self.__params.update(params)
        self.paramStateChanged()

    def paramStateChanged(self):
        if False:
            i = 10
            return i + 15
        pass

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.getParam(item)

    def __len__(self):
        if False:
            print('Hello World!')
        return 0

    def getParam(self, param):
        if False:
            i = 10
            return i + 15
        return self.__params[param]

class Optic(pg.GraphicsObject, ParamObj):
    sigStateChanged = QtCore.Signal()

    def __init__(self, gitem, **params):
        if False:
            print('Hello World!')
        ParamObj.__init__(self)
        pg.GraphicsObject.__init__(self)
        self.gitem = gitem
        self.surfaces = gitem.surfaces
        gitem.setParentItem(self)
        self.roi = pg.ROI([0, 0], [1, 1])
        self.roi.addRotateHandle([1, 1], [0.5, 0.5])
        self.roi.setParentItem(self)
        defaults = {'pos': Point(0, 0), 'angle': 0}
        defaults.update(params)
        self._ior_cache = {}
        self._connRoiChanged = self.roi.sigRegionChanged.connect(self.roiChanged)
        self.setParams(**defaults)

    def updateTransform(self):
        if False:
            for i in range(10):
                print('nop')
        self.setPos(0, 0)
        tr = QtGui.QTransform()
        self.setTransform(tr.translate(Point(self['pos'])).rotate(self['angle']))

    def setParam(self, param, val):
        if False:
            i = 10
            return i + 15
        ParamObj.setParam(self, param, val)

    def paramStateChanged(self):
        if False:
            i = 10
            return i + 15
        'Some parameters of the optic have changed.'
        self.gitem.setPos(Point(self['pos']))
        self.gitem.resetTransform()
        self.gitem.setRotation(self['angle'])
        try:
            if isinstance(self._connRoiChanged, QtCore.QMetaObject.Connection):
                self.roi.sigRegionChanged.disconnect(self._connRoiChanged)
            else:
                self.roi.sigRegionChanged.disconnect(self.roiChanged)
            br = self.gitem.boundingRect()
            o = self.gitem.mapToParent(br.topLeft())
            self.roi.setAngle(self['angle'])
            self.roi.setPos(o)
            self.roi.setSize([br.width(), br.height()])
        finally:
            self._connRoiChanged = self.roi.sigRegionChanged.connect(self.roiChanged)
        self.sigStateChanged.emit()

    def roiChanged(self, *args):
        if False:
            i = 10
            return i + 15
        pos = self.roi.pos()
        self.gitem.resetTransform()
        self.gitem.setRotation(self.roi.angle())
        br = self.gitem.boundingRect()
        o1 = self.gitem.mapToParent(br.topLeft())
        self.setParams(angle=self.roi.angle(), pos=pos + (self.gitem.pos() - o1))

    def boundingRect(self):
        if False:
            print('Hello World!')
        return QtCore.QRectF()

    def paint(self, p, *args):
        if False:
            print('Hello World!')
        pass

    def ior(self, wavelength):
        if False:
            for i in range(10):
                print('nop')
        return GLASSDB.ior(self['glass'], wavelength)

class Lens(Optic):

    def __init__(self, **params):
        if False:
            while True:
                i = 10
        defaults = {'dia': 25.4, 'r1': 50.0, 'r2': 0, 'd': 4.0, 'glass': 'N-BK7', 'reflect': False}
        defaults.update(params)
        d = defaults.pop('d')
        defaults['x1'] = -d / 2.0
        defaults['x2'] = d / 2.0
        gitem = CircularSolid(brush=(100, 100, 130, 100), **defaults)
        Optic.__init__(self, gitem, **defaults)

    def propagateRay(self, ray):
        if False:
            for i in range(10):
                print('nop')
        'Refract, reflect, absorb, and/or scatter ray. This function may create and return new rays'
        '\n        NOTE:: We can probably use this to compute refractions faster: (from GLSL 120 docs)\n\n        For the incident vector I and surface normal N, and the\n        ratio of indices of refraction eta, return the refraction\n        vector. The result is computed by\n        k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I))\n        if (k < 0.0)\n            return genType(0.0)\n        else\n            return eta * I - (eta * dot(N, I) + sqrt(k)) * N\n        The input parameters for the incident vector I and the\n        surface normal N must already be normalized to get the\n        desired results. eta == ratio of IORs\n\n\n        For reflection:\n        For the incident vector I and surface orientation N,\n        returns the reflection direction:\n        I – 2 ∗ dot(N, I) ∗ N\n        N must already be normalized in order to achieve the\n        desired result.\n        '
        iors = [self.ior(ray['wl']), 1.0]
        for i in [0, 1]:
            surface = self.surfaces[i]
            ior = iors[i]
            (p1, ai) = surface.intersectRay(ray)
            if p1 is None:
                ray.setEnd(None)
                break
            p1 = surface.mapToItem(ray, p1)
            rd = ray['dir']
            a1 = atan2(rd[1], rd[0])
            try:
                ar = a1 - ai + asin(sin(ai) * ray['ior'] / ior)
            except ValueError:
                ar = np.nan
            ray.setEnd(p1)
            dp = Point(cos(ar), sin(ar))
            ray = Ray(parent=ray, ior=ior, dir=dp)
        return [ray]

class Mirror(Optic):

    def __init__(self, **params):
        if False:
            return 10
        defaults = {'r1': 0, 'r2': 0, 'd': 0.01}
        defaults.update(params)
        d = defaults.pop('d')
        defaults['x1'] = -d / 2.0
        defaults['x2'] = d / 2.0
        gitem = CircularSolid(brush=(100, 100, 100, 255), **defaults)
        Optic.__init__(self, gitem, **defaults)

    def propagateRay(self, ray):
        if False:
            return 10
        'Refract, reflect, absorb, and/or scatter ray. This function may create and return new rays'
        surface = self.surfaces[0]
        (p1, ai) = surface.intersectRay(ray)
        if p1 is not None:
            p1 = surface.mapToItem(ray, p1)
            rd = ray['dir']
            a1 = atan2(rd[1], rd[0])
            ar = a1 + np.pi - 2 * ai
            ray.setEnd(p1)
            dp = Point(cos(ar), sin(ar))
            ray = Ray(parent=ray, dir=dp)
        else:
            ray.setEnd(None)
        return [ray]

class CircularSolid(pg.GraphicsObject, ParamObj):
    """GraphicsObject with two circular or flat surfaces."""

    def __init__(self, pen=None, brush=None, **opts):
        if False:
            print('Hello World!')
        '\n        Arguments for each surface are:\n           x1,x2 - position of center of _physical surface_\n           r1,r2 - radius of curvature\n           d1,d2 - diameter of optic\n        '
        defaults = dict(x1=-2, r1=100, d1=25.4, x2=2, r2=100, d2=25.4)
        defaults.update(opts)
        ParamObj.__init__(self)
        self.surfaces = [CircleSurface(defaults['r1'], defaults['d1']), CircleSurface(-defaults['r2'], defaults['d2'])]
        pg.GraphicsObject.__init__(self)
        for s in self.surfaces:
            s.setParentItem(self)
        if pen is None:
            self.pen = pg.mkPen((220, 220, 255, 200), width=1, cosmetic=True)
        else:
            self.pen = pg.mkPen(pen)
        if brush is None:
            self.brush = pg.mkBrush((230, 230, 255, 30))
        else:
            self.brush = pg.mkBrush(brush)
        self.setParams(**defaults)

    def paramStateChanged(self):
        if False:
            return 10
        self.updateSurfaces()

    def updateSurfaces(self):
        if False:
            return 10
        self.surfaces[0].setParams(self['r1'], self['d1'])
        self.surfaces[1].setParams(-self['r2'], self['d2'])
        self.surfaces[0].setPos(self['x1'], 0)
        self.surfaces[1].setPos(self['x2'], 0)
        self.path = QtGui.QPainterPath()
        self.path.connectPath(self.surfaces[0].path.translated(self.surfaces[0].pos()))
        self.path.connectPath(self.surfaces[1].path.translated(self.surfaces[1].pos()).toReversed())
        self.path.closeSubpath()

    def boundingRect(self):
        if False:
            for i in range(10):
                print('nop')
        return self.path.boundingRect()

    def shape(self):
        if False:
            print('Hello World!')
        return self.path

    def paint(self, p, *args):
        if False:
            print('Hello World!')
        p.setRenderHints(p.renderHints() | p.RenderHint.Antialiasing)
        p.setPen(self.pen)
        p.fillPath(self.path, self.brush)
        p.drawPath(self.path)

class CircleSurface(pg.GraphicsObject):

    def __init__(self, radius=None, diameter=None):
        if False:
            i = 10
            return i + 15
        "center of physical surface is at 0,0\n        radius is the radius of the surface. If radius is None, the surface is flat. \n        diameter is of the optic's edge."
        pg.GraphicsObject.__init__(self)
        self.r = radius
        self.d = diameter
        self.mkPath()

    def setParams(self, r, d):
        if False:
            i = 10
            return i + 15
        self.r = r
        self.d = d
        self.mkPath()

    def mkPath(self):
        if False:
            print('Hello World!')
        self.prepareGeometryChange()
        r = self.r
        d = self.d
        h2 = d / 2.0
        self.path = QtGui.QPainterPath()
        if r == 0:
            self.path.moveTo(0, h2)
            self.path.lineTo(0, -h2)
        else:
            h2 = min(h2, abs(r))
            arc = QtCore.QRectF(0, -r, r * 2, r * 2)
            a1 = degrees(asin(h2 / r))
            a2 = -2 * a1
            a1 += 180.0
            self.path.arcMoveTo(arc, a1)
            self.path.arcTo(arc, a1, a2)
        self.h2 = h2

    def boundingRect(self):
        if False:
            while True:
                i = 10
        return self.path.boundingRect()

    def paint(self, p, *args):
        if False:
            print('Hello World!')
        return

    def intersectRay(self, ray):
        if False:
            for i in range(10):
                print('nop')
        h = self.h2
        r = self.r
        (p, dir) = ray.currentState(relativeTo=self)
        p = p - Point(r, 0)
        if r == 0:
            if dir[0] == 0:
                y = 0
            else:
                y = p[1] - p[0] * dir[1] / dir[0]
            if abs(y) > h:
                return (None, None)
            else:
                return (Point(0, y), atan2(dir[1], dir[0]))
        else:
            dx = dir[0]
            dy = dir[1]
            dr = hypot(dx, dy)
            D = p[0] * (p[1] + dy) - (p[0] + dx) * p[1]
            idr2 = 1.0 / dr ** 2
            disc = r ** 2 * dr ** 2 - D ** 2
            if disc < 0:
                return (None, None)
            disc2 = disc ** 0.5
            if dy < 0:
                sgn = -1
            else:
                sgn = 1
            br = self.path.boundingRect()
            x1 = (D * dy + sgn * dx * disc2) * idr2
            y1 = (-D * dx + abs(dy) * disc2) * idr2
            if br.contains(x1 + r, y1):
                pt = Point(x1, y1)
            else:
                x2 = (D * dy - sgn * dx * disc2) * idr2
                y2 = (-D * dx - abs(dy) * disc2) * idr2
                pt = Point(x2, y2)
                if not br.contains(x2 + r, y2):
                    return (None, None)
            norm = atan2(pt[1], pt[0])
            if r < 0:
                norm += np.pi
            dp = p - pt
            ang = atan2(dp[1], dp[0])
            return (pt + Point(r, 0), ang - norm)

class Ray(pg.GraphicsObject, ParamObj):
    """Represents a single straight segment of a ray"""
    sigStateChanged = QtCore.Signal()

    def __init__(self, **params):
        if False:
            for i in range(10):
                print('nop')
        ParamObj.__init__(self)
        defaults = {'ior': 1.0, 'wl': 500, 'end': None, 'dir': Point(1, 0)}
        self.params = {}
        pg.GraphicsObject.__init__(self)
        self.children = []
        parent = params.get('parent', None)
        if parent is not None:
            defaults['start'] = parent['end']
            defaults['wl'] = parent['wl']
            self['ior'] = parent['ior']
            self['dir'] = parent['dir']
            parent.addChild(self)
        defaults.update(params)
        defaults['dir'] = Point(defaults['dir'])
        self.setParams(**defaults)
        self.mkPath()

    def clearChildren(self):
        if False:
            return 10
        for c in self.children:
            c.clearChildren()
            c.setParentItem(None)
            self.scene().removeItem(c)
        self.children = []

    def paramStateChanged(self):
        if False:
            return 10
        pass

    def addChild(self, ch):
        if False:
            i = 10
            return i + 15
        self.children.append(ch)
        ch.setParentItem(self)

    def currentState(self, relativeTo=None):
        if False:
            i = 10
            return i + 15
        pos = self['start']
        dir = self['dir']
        if relativeTo is None:
            return (pos, dir)
        else:
            trans = self.itemTransform(relativeTo)[0]
            p1 = trans.map(pos)
            p2 = trans.map(pos + dir)
            return (Point(p1), Point(p2 - p1))

    def setEnd(self, end):
        if False:
            while True:
                i = 10
        self['end'] = end
        self.mkPath()

    def boundingRect(self):
        if False:
            while True:
                i = 10
        return self.path.boundingRect()

    def paint(self, p, *args):
        if False:
            i = 10
            return i + 15
        p.setRenderHints(p.renderHints() | p.RenderHint.Antialiasing)
        p.setCompositionMode(p.CompositionMode.CompositionMode_Plus)
        p.setPen(wlPen(self['wl']))
        p.drawPath(self.path)

    def mkPath(self):
        if False:
            print('Hello World!')
        self.prepareGeometryChange()
        self.path = QtGui.QPainterPath()
        self.path.moveTo(self['start'])
        if self['end'] is not None:
            self.path.lineTo(self['end'])
        else:
            self.path.lineTo(self['start'] + 500 * self['dir'])

def trace(rays, optics):
    if False:
        while True:
            i = 10
    if len(optics) < 1 or len(rays) < 1:
        return
    for r in rays:
        r.clearChildren()
        o = optics[0]
        r2 = o.propagateRay(r)
        trace(r2, optics[1:])

class Tracer(QtCore.QObject):
    """
    Simple ray tracer. 
    
    Initialize with a list of rays and optics; 
    calling trace() will cause rays to be extended by propagating them through
    each optic in sequence.
    """

    def __init__(self, rays, optics):
        if False:
            for i in range(10):
                print('nop')
        QtCore.QObject.__init__(self)
        self.optics = optics
        self.rays = rays
        for o in self.optics:
            o.sigStateChanged.connect(self.trace)
        self.trace()

    def trace(self):
        if False:
            for i in range(10):
                print('nop')
        trace(self.rays, self.optics)