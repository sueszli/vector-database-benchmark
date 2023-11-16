"""OnscreenImage module: contains the OnscreenImage class.

See the :ref:`onscreenimage` page in the programming manual for explanation of
this class.
"""
__all__ = ['OnscreenImage']
from panda3d.core import CardMaker, NodePath, Texture, TexturePool, VBase3
from direct.showbase.DirectObject import DirectObject

class OnscreenImage(DirectObject, NodePath):

    def __init__(self, image=None, pos=None, hpr=None, scale=None, color=None, parent=None, sort=0):
        if False:
            print('Hello World!')
        "\n        Make a image node from string or a `~panda3d.core.NodePath`, put\n        it into the 2-D scene graph and set it up with all the indicated\n        parameters.\n\n        Parameters:\n\n          image: the actual geometry to display or a file name.\n                 This may be omitted and specified later via setImage()\n                 if you don't have it available.\n\n          pos: the x, y, z position of the geometry on the screen.\n               This maybe a 3-tuple of floats or a vector.\n               y should be zero\n\n          hpr: the h, p, r of the geometry on the screen.\n               This maybe a 3-tuple of floats or a vector.\n\n          scale: the size of the geometry.  This may either be a single\n                 float, a 3-tuple of floats, or a vector, specifying a\n                 different x, y, z scale.  y should be 1\n\n          color: the (r, g, b, a) color of the geometry.  This is\n                 normally a 4-tuple of floats or ints.\n\n          parent: the NodePath to parent the geometry to initially.\n        "
        NodePath.__init__(self)
        if parent is None:
            from direct.showbase import ShowBaseGlobal
            parent = ShowBaseGlobal.aspect2d
        self.setImage(image, parent=parent, sort=sort)
        if isinstance(pos, tuple) or isinstance(pos, list):
            self.setPos(*pos)
        elif isinstance(pos, VBase3):
            self.setPos(pos)
        if isinstance(hpr, tuple) or isinstance(hpr, list):
            self.setHpr(*hpr)
        elif isinstance(hpr, VBase3):
            self.setHpr(hpr)
        if isinstance(scale, tuple) or isinstance(scale, list):
            self.setScale(*scale)
        elif isinstance(scale, VBase3):
            self.setScale(scale)
        elif isinstance(scale, float) or isinstance(scale, int):
            self.setScale(scale)
        if color:
            self.setColor(color[0], color[1], color[2], color[3])

    def setImage(self, image, parent=NodePath(), transform=None, sort=0):
        if False:
            i = 10
            return i + 15
        if not self.isEmpty():
            parent = self.getParent()
            if transform is None:
                transform = self.getTransform()
            sort = self.getSort()
        self.removeNode()
        if isinstance(image, NodePath):
            self.assign(image.copyTo(parent, sort))
        elif isinstance(image, str) or isinstance(image, Texture):
            if isinstance(image, Texture):
                tex = image
            else:
                tex = TexturePool.loadTexture(image)
                if not tex:
                    raise IOError('Could not load texture: %s' % image)
            cm = CardMaker('OnscreenImage')
            cm.setFrame(-1, 1, -1, 1)
            self.assign(parent.attachNewNode(cm.generate(), sort))
            self.setTexture(tex)
        elif isinstance(image, tuple):
            model = base.loader.loadModel(image[0])
            if model:
                node = model.find(image[1])
                if node:
                    self.assign(node.copyTo(parent, sort))
                else:
                    print('OnscreenImage: node %s not found' % image[1])
            else:
                print('OnscreenImage: model %s not found' % image[0])
        if transform and (not self.isEmpty()):
            self.setTransform(transform)

    def getImage(self):
        if False:
            return 10
        return self

    def configure(self, option=None, **kw):
        if False:
            while True:
                i = 10
        for (option, value) in kw.items():
            try:
                setter = getattr(self, 'set' + option[0].upper() + option[1:])
                if (setter == self.setPos or setter == self.setHpr or setter == self.setScale) and isinstance(value, (tuple, list)):
                    setter(*value)
                else:
                    setter(value)
            except AttributeError:
                print('OnscreenImage.configure: invalid option: %s' % option)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        self.configure(*(), **{key: value})

    def cget(self, option):
        if False:
            return 10
        getter = getattr(self, 'get' + option[0].upper() + option[1:])
        return getter()
    __getitem__ = cget

    def destroy(self):
        if False:
            print('Hello World!')
        self.removeNode()