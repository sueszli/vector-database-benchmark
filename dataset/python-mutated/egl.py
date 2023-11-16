"""A ctypes-based API to EGL."""
import os
import ctypes
from ctypes import c_int as _c_int, POINTER as _POINTER, c_void_p, c_char_p
_egl_file = None
if 'EGL_LIBRARY' in os.environ:
    if os.path.exists(os.environ['EGL_LIBRARY']):
        _egl_file = os.path.realpath(os.environ['EGL_LIBRARY'])
if _egl_file is None:
    _egl_file = ctypes.util.find_library('EGL')
if _egl_file is None:
    raise OSError('EGL library not found')
_lib = ctypes.CDLL(_egl_file)
EGL_FALSE = 0
EGL_TRUE = 1
EGL_DEFAULT_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_NO_DISPLAY = 0
EGL_NO_SURFACE = 0
EGL_DONT_CARE = -1
EGL_SUCCESS = 12288
EGL_NOT_INITIALIZED = 12289
EGL_BAD_ACCESS = 12290
EGL_BAD_ALLOC = 12291
EGL_BAD_ATTRIBUTE = 12292
EGL_BAD_CONFIG = 12293
EGL_BAD_CONTEXT = 12294
EGL_BAD_CURRENT_SURFACE = 12295
EGL_BAD_DISPLAY = 12296
EGL_BAD_MATCH = 12297
EGL_BAD_NATIVE_PIXMAP = 12298
EGL_BAD_NATIVE_WINDOW = 12299
EGL_BAD_PARAMETER = 12300
EGL_BAD_SURFACE = 12301
EGL_CONTEXT_LOST = 12302
EGL_BUFFER_SIZE = 12320
EGL_ALPHA_SIZE = 12321
EGL_BLUE_SIZE = 12322
EGL_GREEN_SIZE = 12323
EGL_RED_SIZE = 12324
EGL_DEPTH_SIZE = 12325
EGL_STENCIL_SIZE = 12326
EGL_CONFIG_CAVEAT = 12327
EGL_CONFIG_ID = 12328
EGL_LEVEL = 12329
EGL_MAX_PBUFFER_HEIGHT = 12330
EGL_MAX_PBUFFER_PIXELS = 12331
EGL_MAX_PBUFFER_WIDTH = 12332
EGL_NATIVE_RENDERABLE = 12333
EGL_NATIVE_VISUAL_ID = 12334
EGL_NATIVE_VISUAL_TYPE = 12335
EGL_SAMPLES = 12337
EGL_SAMPLE_BUFFERS = 12338
EGL_SURFACE_TYPE = 12339
EGL_TRANSPARENT_TYPE = 12340
EGL_TRANSPARENT_BLUE_VALUE = 12341
EGL_TRANSPARENT_GREEN_VALUE = 12342
EGL_TRANSPARENT_RED_VALUE = 12343
EGL_NONE = 12344
EGL_BIND_TO_TEXTURE_RGB = 12345
EGL_BIND_TO_TEXTURE_RGBA = 12346
EGL_MIN_SWAP_INTERVAL = 12347
EGL_MAX_SWAP_INTERVAL = 12348
EGL_LUMINANCE_SIZE = 12349
EGL_ALPHA_MASK_SIZE = 12350
EGL_COLOR_BUFFER_TYPE = 12351
EGL_RENDERABLE_TYPE = 12352
EGL_MATCH_NATIVE_PIXMAP = 12353
EGL_CONFORMANT = 12354
EGL_SLOW_CONFIG = 12368
EGL_NON_CONFORMANT_CONFIG = 12369
EGL_TRANSPARENT_RGB = 12370
EGL_RGB_BUFFER = 12430
EGL_LUMINANCE_BUFFER = 12431
EGL_NO_TEXTURE = 12380
EGL_TEXTURE_RGB = 12381
EGL_TEXTURE_RGBA = 12382
EGL_TEXTURE_2D = 12383
EGL_PBUFFER_BIT = 1
EGL_PIXMAP_BIT = 2
EGL_WINDOW_BIT = 4
EGL_VG_COLORSPACE_LINEAR_BIT = 32
EGL_VG_ALPHA_FORMAT_PRE_BIT = 64
EGL_MULTISAMPLE_RESOLVE_BOX_BIT = 512
EGL_SWAP_BEHAVIOR_PRESERVED_BIT = 1024
EGL_OPENGL_ES_BIT = 1
EGL_OPENVG_BIT = 2
EGL_OPENGL_ES2_BIT = 4
EGL_OPENGL_BIT = 8
EGL_VENDOR = 12371
EGL_VERSION = 12372
EGL_EXTENSIONS = 12373
EGL_CLIENT_APIS = 12429
EGL_HEIGHT = 12374
EGL_WIDTH = 12375
EGL_LARGEST_PBUFFER = 12376
EGL_TEXTURE_FORMAT = 12416
EGL_TEXTURE_TARGET = 12417
EGL_MIPMAP_TEXTURE = 12418
EGL_MIPMAP_LEVEL = 12419
EGL_RENDER_BUFFER = 12422
EGL_VG_COLORSPACE = 12423
EGL_VG_ALPHA_FORMAT = 12424
EGL_HORIZONTAL_RESOLUTION = 12432
EGL_VERTICAL_RESOLUTION = 12433
EGL_PIXEL_ASPECT_RATIO = 12434
EGL_SWAP_BEHAVIOR = 12435
EGL_MULTISAMPLE_RESOLVE = 12441
EGL_BACK_BUFFER = 12420
EGL_SINGLE_BUFFER = 12421
EGL_VG_COLORSPACE_sRGB = 12425
EGL_VG_COLORSPACE_LINEAR = 12426
EGL_VG_ALPHA_FORMAT_NONPRE = 12427
EGL_VG_ALPHA_FORMAT_PRE = 12428
EGL_DISPLAY_SCALING = 10000
EGL_UNKNOWN = -1
EGL_BUFFER_PRESERVED = 12436
EGL_BUFFER_DESTROYED = 12437
EGL_OPENVG_IMAGE = 12438
EGL_CONTEXT_CLIENT_TYPE = 12439
EGL_CONTEXT_CLIENT_VERSION = 12440
EGL_MULTISAMPLE_RESOLVE_DEFAULT = 12442
EGL_MULTISAMPLE_RESOLVE_BOX = 12443
EGL_OPENGL_ES_API = 12448
EGL_OPENVG_API = 12449
EGL_OPENGL_API = 12450
EGL_DRAW = 12377
EGL_READ = 12378
EGL_CORE_NATIVE_ENGINE = 12379
EGL_COLORSPACE = EGL_VG_COLORSPACE
EGL_ALPHA_FORMAT = EGL_VG_ALPHA_FORMAT
EGL_COLORSPACE_sRGB = EGL_VG_COLORSPACE_sRGB
EGL_COLORSPACE_LINEAR = EGL_VG_COLORSPACE_LINEAR
EGL_ALPHA_FORMAT_NONPRE = EGL_VG_ALPHA_FORMAT_NONPRE
EGL_ALPHA_FORMAT_PRE = EGL_VG_ALPHA_FORMAT_PRE
_lib.eglGetDisplay.argtypes = (_c_int,)
_lib.eglGetDisplay.restype = c_void_p
_lib.eglInitialize.argtypes = (c_void_p, _POINTER(_c_int), _POINTER(_c_int))
_lib.eglTerminate.argtypes = (c_void_p,)
_lib.eglChooseConfig.argtypes = (c_void_p, _POINTER(_c_int), _POINTER(c_void_p), _c_int, _POINTER(_c_int))
_lib.eglCreateWindowSurface.argtypes = (c_void_p, c_void_p, c_void_p, _POINTER(_c_int))
_lib.eglCreateWindowSurface.restype = c_void_p
_lib.eglCreatePbufferSurface.argtypes = (c_void_p, c_void_p, _POINTER(_c_int))
_lib.eglCreatePbufferSurface.restype = c_void_p
_lib.eglCreateContext.argtypes = (c_void_p, c_void_p, c_void_p, _POINTER(_c_int))
_lib.eglCreateContext.restype = c_void_p
_lib.eglMakeCurrent.argtypes = (c_void_p,) * 4
_lib.eglBindAPI.argtypes = (_c_int,)
_lib.eglSwapBuffers.argtypes = (c_void_p,) * 2
_lib.eglDestroySurface.argtypes = (c_void_p,) * 2
_lib.eglQueryString.argtypes = (c_void_p, _c_int)
_lib.eglQueryString.restype = c_char_p

def eglGetError():
    if False:
        for i in range(10):
            print('nop')
    'Check for errors, returns an enum (int).'
    return _lib.eglGetError()

def eglGetDisplay(display=EGL_DEFAULT_DISPLAY):
    if False:
        i = 10
        return i + 15
    'Connect to the EGL display server.'
    res = _lib.eglGetDisplay(display)
    if not res or res == EGL_NO_DISPLAY:
        raise RuntimeError('Could not create display')
    return c_void_p(res)

def eglInitialize(display):
    if False:
        return 10
    'Initialize EGL and return EGL version tuple.'
    majorVersion = (_c_int * 1)()
    minorVersion = (_c_int * 1)()
    res = _lib.eglInitialize(display, majorVersion, minorVersion)
    if res == EGL_FALSE:
        raise RuntimeError('Could not initialize')
    return (majorVersion[0], minorVersion[0])

def eglTerminate(display):
    if False:
        print('Hello World!')
    'Terminate an EGL display connection.'
    _lib.eglTerminate(display)

def eglQueryString(display, name):
    if False:
        i = 10
        return i + 15
    'Query string from display'
    out = _lib.eglQueryString(display, name)
    if not out:
        raise RuntimeError('Could not query %s' % name)
    return out
DEFAULT_ATTRIB_LIST = (EGL_RED_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_BIND_TO_TEXTURE_RGBA, EGL_TRUE, EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER, EGL_CONFORMANT, EGL_OPENGL_ES2_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NATIVE_RENDERABLE, EGL_TRUE, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT)

def _convert_attrib_list(attribList):
    if False:
        return 10
    attribList = attribList or []
    attribList = [a for a in attribList] + [EGL_NONE]
    attribList = (_c_int * len(attribList))(*attribList)
    return attribList

def eglChooseConfig(display, attribList=DEFAULT_ATTRIB_LIST):
    if False:
        return 10
    attribList = _convert_attrib_list(attribList)
    numConfigs = (_c_int * 1)()
    _lib.eglChooseConfig(display, attribList, None, 0, numConfigs)
    n = numConfigs[0]
    if n <= 0:
        raise RuntimeError('Could not find any suitable config.')
    config = (c_void_p * n)()
    _lib.eglChooseConfig(display, attribList, config, n, numConfigs)
    return config

def _check_res(res):
    if False:
        i = 10
        return i + 15
    if res == EGL_NO_SURFACE:
        e = eglGetError()
    else:
        return res
    if e == EGL_BAD_MATCH:
        raise ValueError('Cannot create surface: attributes do not match ' + 'or given config cannot render in window.')
    elif e == EGL_BAD_CONFIG:
        raise ValueError('Cannot create surface: given config is not ' + 'supported by this system.')
    elif e == EGL_BAD_NATIVE_WINDOW:
        raise ValueError('Cannot create surface: the given native window ' + 'handle is invalid.')
    elif e == EGL_BAD_ALLOC:
        raise RuntimeError('Could not allocate surface: not enough ' + 'resources or native window already associated ' + 'with another config.')
    else:
        raise RuntimeError('Could not create window surface due to ' + 'unknown error: %i' % e)

def eglCreateWindowSurface(display, config, window, attribList=None):
    if False:
        return 10
    attribList = _convert_attrib_list(attribList)
    surface = c_void_p(_lib.eglCreateWindowSurface(display, config, window, attribList))
    return _check_res(surface)

def eglCreatePbufferSurface(display, config, attribList=None):
    if False:
        i = 10
        return i + 15
    attribList = _convert_attrib_list(attribList)
    surface = c_void_p(_lib.eglCreatePbufferSurface(display, config, attribList))
    return _check_res(surface)

def eglCreateContext(display, config, shareContext=EGL_NO_CONTEXT, attribList=None):
    if False:
        return 10
    attribList = attribList or [EGL_CONTEXT_CLIENT_VERSION, 2]
    attribList = [a for a in attribList] + [EGL_NONE]
    attribList = (_c_int * len(attribList))(*attribList)
    res = c_void_p(_lib.eglCreateContext(display, config, shareContext, attribList))
    if res == EGL_NO_CONTEXT:
        e = eglGetError()
        if e == EGL_BAD_CONFIG:
            raise ValueError('Could not create context: given config is ' + 'not supported by this system.')
        else:
            raise RuntimeError('Could not create context due to ' + 'unknown error: %i' % e)
    return res

def eglMakeCurrent(display, draw, read, context):
    if False:
        for i in range(10):
            print('nop')
    res = _lib.eglMakeCurrent(display, draw, read, context)
    if res == EGL_FALSE:
        raise RuntimeError('Could not make the context current.')

def eglBindAPI(api):
    if False:
        i = 10
        return i + 15
    'Set the current rendering API (OpenGL, OpenGL ES or OpenVG)'
    res = _lib.eglBindAPI(api)
    if res == EGL_FALSE:
        raise RuntimeError('Could not bind API %d' % api)
    return res

def eglSwapBuffers(display, surface):
    if False:
        for i in range(10):
            print('nop')
    res = _lib.eglSwapBuffers(display, surface)
    if res == EGL_FALSE:
        raise RuntimeError('Could not swap buffers.')

def eglDestroySurface(display, surface):
    if False:
        return 10
    res = _lib.eglDestroySurface(display, surface)
    if res == EGL_FALSE:
        raise RuntimeError('Could not destroy surface')