from OpenGL.GL import *
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
GLOptions = {'opaque': {GL_DEPTH_TEST: True, GL_BLEND: False, GL_ALPHA_TEST: False, GL_CULL_FACE: False}, 'translucent': {GL_DEPTH_TEST: True, GL_BLEND: True, GL_ALPHA_TEST: False, GL_CULL_FACE: False, 'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)}, 'additive': {GL_DEPTH_TEST: False, GL_BLEND: True, GL_ALPHA_TEST: False, GL_CULL_FACE: False, 'glBlendFunc': (GL_SRC_ALPHA, GL_ONE)}}

class GLGraphicsItem(QtCore.QObject):
    _nextId = 0

    def __init__(self, parentItem: 'GLGraphicsItem'=None):
        if False:
            return 10
        super().__init__()
        self._id = GLGraphicsItem._nextId
        GLGraphicsItem._nextId += 1
        self.__parent: GLGraphicsItem | None = None
        self.__view = None
        self.__children: set[GLGraphicsItem] = set()
        self.__transform = Transform3D()
        self.__visible = True
        self.__initialized = False
        self.setParentItem(parentItem)
        self.setDepthValue(0)
        self.__glOpts = {}

    def setParentItem(self, item):
        if False:
            return 10
        "Set this item's parent in the scenegraph hierarchy."
        if self.__parent is not None:
            self.__parent.__children.remove(self)
        if item is not None:
            item.__children.add(self)
        self.__parent = item
        if self.__parent is not None and self.view() is not self.__parent.view():
            if self.view() is not None:
                self.view().removeItem(self)
            self.__parent.view().addItem(self)

    def setGLOptions(self, opts):
        if False:
            return 10
        "\n        Set the OpenGL state options to use immediately before drawing this item.\n        (Note that subclasses must call setupGLState before painting for this to work)\n        \n        The simplest way to invoke this method is to pass in the name of\n        a predefined set of options (see the GLOptions variable):\n        \n        ============= ======================================================\n        opaque        Enables depth testing and disables blending\n        translucent   Enables depth testing and blending\n                      Elements must be drawn sorted back-to-front for\n                      translucency to work correctly.\n        additive      Disables depth testing, enables blending.\n                      Colors are added together, so sorting is not required.\n        ============= ======================================================\n        \n        It is also possible to specify any arbitrary settings as a dictionary. \n        This may consist of {'functionName': (args...)} pairs where functionName must \n        be a callable attribute of OpenGL.GL, or {GL_STATE_VAR: bool} pairs \n        which will be interpreted as calls to glEnable or glDisable(GL_STATE_VAR).\n        \n        For example::\n            \n            {\n                GL_ALPHA_TEST: True,\n                GL_CULL_FACE: False,\n                'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),\n            }\n            \n        \n        "
        if isinstance(opts, str):
            opts = GLOptions[opts]
        self.__glOpts = opts.copy()
        self.update()

    def updateGLOptions(self, opts):
        if False:
            for i in range(10):
                print('nop')
        '\n        Modify the OpenGL state options to use immediately before drawing this item.\n        *opts* must be a dictionary as specified by setGLOptions.\n        Values may also be None, in which case the key will be ignored.\n        '
        self.__glOpts.update(opts)

    def parentItem(self):
        if False:
            i = 10
            return i + 15
        "Return a this item's parent in the scenegraph hierarchy."
        return self.__parent

    def childItems(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a list of this item's children in the scenegraph hierarchy."
        return list(self.__children)

    def _setView(self, v):
        if False:
            print('Hello World!')
        self.__view = v

    def view(self):
        if False:
            print('Hello World!')
        return self.__view

    def setDepthValue(self, value):
        if False:
            print('Hello World!')
        '\n        Sets the depth value of this item. Default is 0.\n        This controls the order in which items are drawn--those with a greater depth value will be drawn later.\n        Items with negative depth values are drawn before their parent.\n        (This is analogous to QGraphicsItem.zValue)\n        The depthValue does NOT affect the position of the item or the values it imparts to the GL depth buffer.\n        '
        self.__depthValue = value

    def depthValue(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the depth value of this item. See setDepthValue for more information.'
        return self.__depthValue

    def setTransform(self, tr):
        if False:
            return 10
        "Set the local transform for this object.\n\n        Parameters\n        ----------\n        tr : pyqtgraph.Transform3D\n            Tranformation from the local coordinate system to the parent's.\n        "
        self.__transform = Transform3D(tr)
        self.update()

    def resetTransform(self):
        if False:
            i = 10
            return i + 15
        "Reset this item's transform to an identity transformation."
        self.__transform.setToIdentity()
        self.update()

    def applyTransform(self, tr, local):
        if False:
            while True:
                i = 10
        "\n        Multiply this object's transform by *tr*. \n        If local is True, then *tr* is multiplied on the right of the current transform::\n        \n            newTransform = transform * tr\n            \n        If local is False, then *tr* is instead multiplied on the left::\n        \n            newTransform = tr * transform\n        "
        if local:
            self.setTransform(self.transform() * tr)
        else:
            self.setTransform(tr * self.transform())

    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        "Return this item's transform object."
        return self.__transform

    def viewTransform(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the transform mapping this item's local coordinate system to the \n        view coordinate system."
        tr = self.__transform
        p = self
        while True:
            p = p.parentItem()
            if p is None:
                break
            tr = p.transform() * tr
        return Transform3D(tr)

    def translate(self, dx, dy, dz, local=False):
        if False:
            print('Hello World!')
        "\n        Translate the object by (*dx*, *dy*, *dz*) in its parent's coordinate system.\n        If *local* is True, then translation takes place in local coordinates.\n        "
        tr = Transform3D()
        tr.translate(dx, dy, dz)
        self.applyTransform(tr, local=local)

    def rotate(self, angle, x, y, z, local=False):
        if False:
            while True:
                i = 10
        '\n        Rotate the object around the axis specified by (x,y,z).\n        *angle* is in degrees.\n        \n        '
        tr = Transform3D()
        tr.rotate(angle, x, y, z)
        self.applyTransform(tr, local=local)

    def scale(self, x, y, z, local=True):
        if False:
            i = 10
            return i + 15
        "\n        Scale the object by (*dx*, *dy*, *dz*) in its local coordinate system.\n        If *local* is False, then scale takes place in the parent's coordinates.\n        "
        tr = Transform3D()
        tr.scale(x, y, z)
        self.applyTransform(tr, local=local)

    def hide(self):
        if False:
            i = 10
            return i + 15
        'Hide this item. \n        This is equivalent to setVisible(False).'
        self.setVisible(False)

    def show(self):
        if False:
            i = 10
            return i + 15
        'Make this item visible if it was previously hidden.\n        This is equivalent to setVisible(True).'
        self.setVisible(True)

    def setVisible(self, vis):
        if False:
            print('Hello World!')
        'Set the visibility of this item.'
        self.__visible = vis
        self.update()

    def visible(self):
        if False:
            return 10
        'Return True if the item is currently set to be visible.\n        Note that this does not guarantee that the item actually appears in the\n        view, as it may be obscured or outside of the current view area.'
        return self.__visible

    def initialize(self):
        if False:
            return 10
        self.initializeGL()
        self.__initialized = True

    def isInitialized(self):
        if False:
            while True:
                i = 10
        return self.__initialized

    def initializeGL(self):
        if False:
            while True:
                i = 10
        "\n        Called after an item is added to a GLViewWidget. \n        The widget's GL context is made current before this method is called.\n        (So this would be an appropriate time to generate lists, upload textures, etc.)\n        "
        pass

    def setupGLState(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method is responsible for preparing the GL state options needed to render \n        this item (blending, depth testing, etc). The method is called immediately before painting the item.\n        '
        for (k, v) in self.__glOpts.items():
            if v is None:
                continue
            if isinstance(k, str):
                func = getattr(GL, k)
                func(*v)
            elif v is True:
                glEnable(k)
            else:
                glDisable(k)

    def paint(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by the GLViewWidget to draw this item.\n        It is the responsibility of the item to set up its own modelview matrix,\n        but the caller will take care of pushing/popping.\n        '
        self.setupGLState()

    def update(self):
        if False:
            i = 10
            return i + 15
        '\n        Indicates that this item needs to be redrawn, and schedules an update \n        with the view it is displayed in.\n        '
        v = self.view()
        if v is None:
            return
        v.update()

    def mapToParent(self, point):
        if False:
            i = 10
            return i + 15
        tr = self.transform()
        if tr is None:
            return point
        return tr.map(point)

    def mapFromParent(self, point):
        if False:
            i = 10
            return i + 15
        tr = self.transform()
        if tr is None:
            return point
        return tr.inverted()[0].map(point)

    def mapToView(self, point):
        if False:
            i = 10
            return i + 15
        tr = self.viewTransform()
        if tr is None:
            return point
        return tr.map(point)

    def mapFromView(self, point):
        if False:
            print('Hello World!')
        tr = self.viewTransform()
        if tr is None:
            return point
        return tr.inverted()[0].map(point)