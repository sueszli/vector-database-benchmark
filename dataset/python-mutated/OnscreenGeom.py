"""OnscreenGeom module: contains the OnscreenGeom class"""
__all__ = ['OnscreenGeom']
from panda3d.core import NodePath, VBase3
from direct.showbase.DirectObject import DirectObject

class OnscreenGeom(DirectObject, NodePath):

    def __init__(self, geom=None, pos=None, hpr=None, scale=None, color=None, parent=None, sort=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make a geom node from string or a node path,\n        put it into the 2d sg and set it up with all the indicated parameters.\n\n        The parameters are as follows:\n\n          geom: the actual geometry to display or a file name.\n                This may be omitted and specified later via setGeom()\n                if you don't have it available.\n\n          pos: the x, y, z position of the geometry on the screen.\n               This maybe a 3-tuple of floats or a vector.\n               y should be zero\n\n          hpr: the h, p, r of the geometry on the screen.\n               This maybe a 3-tuple of floats or a vector.\n\n          scale: the size of the geometry.  This may either be a single\n                 float, a 3-tuple of floats, or a vector, specifying a\n                 different x, y, z scale.  y should be 1\n\n          color: the (r, g, b, a) color of the geometry.  This is\n                 normally a 4-tuple of floats or ints.\n\n          parent: the NodePath to parent the geometry to initially.\n        "
        NodePath.__init__(self)
        if parent is None:
            from direct.showbase import ShowBaseGlobal
            parent = ShowBaseGlobal.aspect2d
        self.setGeom(geom, parent=parent, sort=sort, color=color)
        if isinstance(pos, tuple) or isinstance(pos, list):
            self.setPos(*pos)
        elif isinstance(pos, VBase3):
            self.setPos(pos)
        if isinstance(hpr, tuple) or isinstance(hpr, list):
            self.setHpr(*hpr)
        elif isinstance(hpr, VBase3):
            self.setPos(hpr)
        if isinstance(scale, tuple) or isinstance(scale, list):
            self.setScale(*scale)
        elif isinstance(scale, VBase3):
            self.setPos(scale)
        elif isinstance(scale, float) or isinstance(scale, int):
            self.setScale(scale)

    def setGeom(self, geom, parent=NodePath(), transform=None, sort=0, color=None):
        if False:
            return 10
        if not self.isEmpty():
            parent = self.getParent()
            if transform is None:
                transform = self.getTransform()
            sort = self.getSort()
            if color is None and self.hasColor():
                color = self.getColor()
        self.removeNode()
        if isinstance(geom, NodePath):
            self.assign(geom.copyTo(parent, sort))
        elif isinstance(geom, str):
            self.assign(base.loader.loadModel(geom))
            self.reparentTo(parent, sort)
        if not self.isEmpty():
            if transform:
                self.setTransform(transform.compose(self.getTransform()))
            if color:
                self.setColor(color[0], color[1], color[2], color[3])

    def getGeom(self):
        if False:
            i = 10
            return i + 15
        return self

    def configure(self, option=None, **kw):
        if False:
            i = 10
            return i + 15
        for (option, value) in kw.items():
            try:
                setter = getattr(self, 'set' + option[0].upper() + option[1:])
                if (setter == self.setPos or setter == self.setHpr or setter == self.setScale) and (isinstance(value, tuple) or isinstance(value, list)):
                    setter(*value)
                else:
                    setter(value)
            except AttributeError:
                print('OnscreenText.configure: invalid option: %s' % option)

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.configure(*(), **{key: value})

    def cget(self, option):
        if False:
            while True:
                i = 10
        getter = getattr(self, 'get' + option[0].upper() + option[1:])
        return getter()
    __getitem__ = cget

    def destroy(self):
        if False:
            print('Hello World!')
        self.removeNode()