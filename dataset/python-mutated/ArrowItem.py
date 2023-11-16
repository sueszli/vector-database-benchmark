from math import hypot
from .. import functions as fn
from ..Qt import QtGui, QtWidgets
__all__ = ['ArrowItem']

class ArrowItem(QtWidgets.QGraphicsPathItem):
    """
    For displaying scale-invariant arrows.
    For arrows pointing to a location on a curve, see CurveArrow
    
    """

    def __init__(self, parent=None, **opts):
        if False:
            return 10
        '\n        Arrows can be initialized with any keyword arguments accepted by \n        the setStyle() method.\n        '
        self.opts = {}
        QtWidgets.QGraphicsPathItem.__init__(self, parent)
        if 'size' in opts:
            opts['headLen'] = opts['size']
        if 'width' in opts:
            opts['headWidth'] = opts['width']
        pos = opts.pop('pos', (0, 0))
        defaultOpts = {'pxMode': True, 'angle': -150, 'headLen': 20, 'headWidth': None, 'tipAngle': 25, 'baseAngle': 0, 'tailLen': None, 'tailWidth': 3, 'pen': (200, 200, 200), 'brush': (50, 50, 200)}
        defaultOpts.update(opts)
        self.setStyle(**defaultOpts)
        self.setPos(*pos)

    def setStyle(self, **opts):
        if False:
            print('Hello World!')
        "\n        Changes the appearance of the arrow.\n        All arguments are optional:\n        \n        ======================  =================================================\n        **Keyword Arguments:**\n        angle                   Orientation of the arrow in degrees. Default is\n                                0; arrow pointing to the left.\n        headLen                 Length of the arrow head, from tip to base.\n                                default=20\n        headWidth               Width of the arrow head at its base. If\n                                headWidth is specified, it overrides tipAngle.\n        tipAngle                Angle of the tip of the arrow in degrees. Smaller\n                                values make a 'sharper' arrow. default=25\n        baseAngle               Angle of the base of the arrow head. Default is\n                                0, which means that the base of the arrow head\n                                is perpendicular to the arrow tail.\n        tailLen                 Length of the arrow tail, measured from the base\n                                of the arrow head to the end of the tail. If\n                                this value is None, no tail will be drawn.\n                                default=None\n        tailWidth               Width of the tail. default=3\n        pen                     The pen used to draw the outline of the arrow.\n        brush                   The brush used to fill the arrow.\n        pxMode                  If True, then the arrow is drawn as a fixed size\n                                regardless of the scale of its parents (including\n                                the ViewBox zoom level). \n        ======================  =================================================\n        "
        arrowOpts = ['headLen', 'tipAngle', 'baseAngle', 'tailLen', 'tailWidth', 'headWidth']
        allowedOpts = ['angle', 'pen', 'brush', 'pxMode'] + arrowOpts
        needUpdate = False
        for (k, v) in opts.items():
            if k not in allowedOpts:
                raise KeyError('Invalid arrow style option "%s"' % k)
            if self.opts.get(k) != v:
                needUpdate = True
            self.opts[k] = v
        if not needUpdate:
            return
        opt = dict([(k, self.opts[k]) for k in arrowOpts if k in self.opts])
        tr = QtGui.QTransform()
        tr.rotate(self.opts['angle'])
        self.path = tr.map(fn.makeArrowPath(**opt))
        self.setPath(self.path)
        self.setPen(fn.mkPen(self.opts['pen']))
        self.setBrush(fn.mkBrush(self.opts['brush']))
        if self.opts['pxMode']:
            self.setFlags(self.flags() | self.GraphicsItemFlag.ItemIgnoresTransformations)
        else:
            self.setFlags(self.flags() & ~self.GraphicsItemFlag.ItemIgnoresTransformations)

    def paint(self, p, *args):
        if False:
            while True:
                i = 10
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        super().paint(p, *args)

    def shape(self):
        if False:
            print('Hello World!')
        return self.path

    def dataBounds(self, ax, frac, orthoRange=None):
        if False:
            print('Hello World!')
        pw = 0
        pen = self.pen()
        if not pen.isCosmetic():
            pw = pen.width() * 0.7072
        if self.opts['pxMode']:
            return [0, 0]
        else:
            br = self.boundingRect()
            if ax == 0:
                return [br.left() - pw, br.right() + pw]
            else:
                return [br.top() - pw, br.bottom() + pw]

    def pixelPadding(self):
        if False:
            return 10
        pad = 0
        if self.opts['pxMode']:
            br = self.boundingRect()
            pad += hypot(br.width(), br.height())
        pen = self.pen()
        if pen.isCosmetic():
            pad += max(1, pen.width()) * 0.7072
        return pad