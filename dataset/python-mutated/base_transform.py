"""
API Issues to work out:

  - MatrixTransform and STTransform both have 'scale' and 'translate'
    attributes, but they are used in very different ways. It would be nice
    to keep this consistent, but how?

  - Need a transform.map_rect function that returns the bounding rectangle of
    a rect after transformation. Non-linear transforms might need to work
    harder at this, but we can provide a default implementation that
    works by mapping a selection of points across a grid within the original
    rect.
"""
from __future__ import division
from ..shaders import Function
from ...util.event import EventEmitter

class BaseTransform(object):
    """
    BaseTransform is a base class that defines a pair of complementary
    coordinate mapping functions in both python and GLSL.

    All BaseTransform subclasses define map() and imap() methods that map
    an object through the forward or inverse transformation, respectively.

    The two class variables glsl_map and glsl_imap are instances of
    shaders.Function that define the forward- and inverse-mapping GLSL
    function code.

    Optionally, an inverse() method returns a new transform performing the
    inverse mapping.

    Note that although all classes should define both map() and imap(), it
    is not necessarily the case that imap(map(x)) == x; there may be instances
    where the inverse mapping is ambiguous or otherwise meaningless.
    """
    glsl_map = None
    glsl_imap = None
    Linear = None
    Orthogonal = None
    NonScaling = None
    Isometric = None

    def __init__(self):
        if False:
            return 10
        self._inverse = None
        self._dynamic = False
        self.changed = EventEmitter(source=self, type='transform_changed')
        if self.glsl_map is not None:
            self._shader_map = Function(self.glsl_map)
        if self.glsl_imap is not None:
            self._shader_imap = Function(self.glsl_imap)

    def map(self, obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return *obj* mapped through the forward transformation.\n\n        Parameters\n        ----------\n            obj : tuple (x,y) or (x,y,z)\n                  array with shape (..., 2) or (..., 3)\n        '
        raise NotImplementedError()

    def imap(self, obj):
        if False:
            i = 10
            return i + 15
        '\n        Return *obj* mapped through the inverse transformation.\n\n        Parameters\n        ----------\n            obj : tuple (x,y) or (x,y,z)\n                  array with shape (..., 2) or (..., 3)\n        '
        raise NotImplementedError()

    @property
    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'The inverse of this transform.'
        if self._inverse is None:
            self._inverse = InverseTransform(self)
        return self._inverse

    @property
    def dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        'Boolean flag that indicates whether this transform is expected to \n        change frequently.\n\n        Transforms that are flagged as dynamic will not be collapsed in \n        ``ChainTransform.simplified``. This allows changes to the transform\n        to propagate through the chain without requiring the chain to be\n        re-simplified.\n        '
        return self._dynamic

    @dynamic.setter
    def dynamic(self, d):
        if False:
            return 10
        self._dynamic = d

    def shader_map(self):
        if False:
            return 10
        '\n        Return a shader Function that accepts only a single vec4 argument\n        and defines new attributes / uniforms supplying the Function with\n        any static input.\n        '
        return self._shader_map

    def shader_imap(self):
        if False:
            while True:
                i = 10
        'See shader_map.'
        return self._shader_imap

    def _shader_object(self):
        if False:
            i = 10
            return i + 15
        "This method allows transforms to be assigned directly to shader\n        template variables. \n\n        Example::\n\n            code = 'void main() { gl_Position = $transform($position); }'\n            func = shaders.Function(code)\n            tr = STTransform()\n            func['transform'] = tr  # use tr's forward mapping for $function\n        "
        return self.shader_map()

    def update(self, *args):
        if False:
            return 10
        'Called to inform any listeners that this transform has changed.'
        self.changed(*args)

    def __mul__(self, tr):
        if False:
            while True:
                i = 10
        '\n        Transform multiplication returns a new transform that is equivalent to\n        the two operands performed in series.\n\n        By default, multiplying two Transforms `A * B` will return\n        ChainTransform([A, B]). Subclasses may redefine this operation to\n        return more optimized results.\n\n        To ensure that both operands have a chance to simplify the operation,\n        all subclasses should follow the same procedure. For `A * B`:\n\n        1. A.__mul__(B) attempts to generate an optimized transform product.\n        2. If that fails, it must:\n\n               * return super(A).__mul__(B) OR\n               * return NotImplemented if the superclass would return an\n                 invalid result.\n\n        3. When BaseTransform.__mul__(A, B) is called, it returns \n           NotImplemented, which causes B.__rmul__(A) to be invoked.\n        4. B.__rmul__(A) attempts to generate an optimized transform product.\n        5. If that fails, it must:\n\n               * return super(B).__rmul__(A) OR\n               * return ChainTransform([B, A]) if the superclass would return\n                 an invalid result.\n\n        6. When BaseTransform.__rmul__(B, A) is called, ChainTransform([A, B])\n           is returned.\n        '
        return tr.__rmul__(self)

    def __rmul__(self, tr):
        if False:
            for i in range(10):
                print('nop')
        return ChainTransform([tr, self])

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s at 0x%x>' % (self.__class__.__name__, id(self))

    def __del__(self):
        if False:
            return 10
        self.changed.disconnect()

class InverseTransform(BaseTransform):

    def __init__(self, transform):
        if False:
            while True:
                i = 10
        BaseTransform.__init__(self)
        self._inverse = transform
        self.map = transform.imap
        self.imap = transform.map

    @property
    def Linear(self):
        if False:
            while True:
                i = 10
        return self._inverse.Linear

    @property
    def Orthogonal(self):
        if False:
            while True:
                i = 10
        return self._inverse.Orthogonal

    @property
    def NonScaling(self):
        if False:
            return 10
        return self._inverse.NonScaling

    @property
    def Isometric(self):
        if False:
            print('Hello World!')
        return self._inverse.Isometric

    @property
    def shader_map(self):
        if False:
            i = 10
            return i + 15
        return self._inverse.shader_imap

    @property
    def shader_imap(self):
        if False:
            while True:
                i = 10
        return self._inverse.shader_map

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Inverse of %r>' % repr(self._inverse)
from .chain import ChainTransform