"""GL ES 2.0 API implemented via pyOpenGL library. Intended as a
fallback and for testing.
"""
from OpenGL import GL as _GL
import OpenGL.GL.framebufferobjects as _FBO
from ...util import logger
from . import _copy_gl_functions
from ._constants import *

def _patch():
    if False:
        print('Hello World!')
    'Monkey-patch pyopengl to fix a bug in glBufferSubData.'
    import sys
    from OpenGL import GL
    if sys.version_info > (3,):
        buffersubdatafunc = GL.glBufferSubData
        if hasattr(buffersubdatafunc, 'wrapperFunction'):
            buffersubdatafunc = buffersubdatafunc.wrapperFunction
        _m = sys.modules[buffersubdatafunc.__module__]
        _m.long = int
    try:
        from OpenGL.GL.VERSION import GL_2_0
        GL_2_0.GL_OBJECT_SHADER_SOURCE_LENGTH = GL_2_0.GL_SHADER_SOURCE_LENGTH
    except Exception:
        pass
_patch()

def _make_unavailable_func(funcname):
    if False:
        i = 10
        return i + 15

    def cb(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('OpenGL API call "%s" is not available.' % funcname)
    return cb

def _get_function_from_pyopengl(funcname):
    if False:
        for i in range(10):
            print('nop')
    'Try getting the given function from PyOpenGL, return\n    a dummy function (that shows a warning when called) if it\n    could not be found.\n    '
    func = None
    try:
        func = getattr(_GL, funcname)
    except AttributeError:
        try:
            func = getattr(_FBO, funcname)
        except AttributeError:
            func = None
    if not bool(func):
        if funcname.endswith('f'):
            try:
                func = getattr(_GL, funcname[:-1])
            except AttributeError:
                pass
    if func is None:
        func = _make_unavailable_func(funcname)
        logger.warning('warning: %s not available' % funcname)
    return func

def _inject():
    if False:
        i = 10
        return i + 15
    'Copy functions from OpenGL.GL into _pyopengl namespace.'
    NS = _pyopengl2.__dict__
    for (glname, ourname) in _pyopengl2._functions_to_import:
        func = _get_function_from_pyopengl(glname)
        NS[ourname] = func
from . import _pyopengl2
_inject()
_copy_gl_functions(_pyopengl2, globals())