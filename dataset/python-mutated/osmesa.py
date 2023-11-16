"""A ctypes-based API to OSMesa"""
from __future__ import print_function
import os
import ctypes
import ctypes.util
from ctypes import c_int as _c_int, c_uint as _c_uint, c_void_p
GL_RGBA = 6408
GL_UNSIGNED_BYTE = 5121
GL_VERSION = 7938
_osmesa_file = None
if 'OSMESA_LIBRARY' in os.environ:
    if os.path.exists(os.environ['OSMESA_LIBRARY']):
        _osmesa_file = os.path.realpath(os.environ['OSMESA_LIBRARY'])
if _osmesa_file is None:
    _osmesa_file = ctypes.util.find_library('OSMesa')
if _osmesa_file is None:
    raise OSError('OSMesa library not found')
_lib = ctypes.CDLL(_osmesa_file)
OSMESA_RGBA = GL_RGBA
_lib.OSMesaCreateContext.argtypes = (_c_int, c_void_p)
_lib.OSMesaCreateContext.restype = c_void_p
_lib.OSMesaDestroyContext.argtypes = (c_void_p,)
_lib.OSMesaMakeCurrent.argtypes = (c_void_p, c_void_p, _c_int, _c_int, _c_int)
_lib.OSMesaMakeCurrent.restype = _c_int
_lib.OSMesaGetCurrentContext.restype = c_void_p

def allocate_pixels_buffer(width, height):
    if False:
        i = 10
        return i + 15
    'Helper function to allocate a buffer to contain an image of\n    width * height suitable for OSMesaMakeCurrent\n    '
    return (_c_uint * width * height * 4)()

def OSMesaCreateContext():
    if False:
        return 10
    return ctypes.cast(_lib.OSMesaCreateContext(OSMESA_RGBA, None), c_void_p)

def OSMesaDestroyContext(context):
    if False:
        return 10
    _lib.OSMesaDestroyContext(context)

def OSMesaMakeCurrent(context, buffer, width, height):
    if False:
        while True:
            i = 10
    ret = _lib.OSMesaMakeCurrent(context, buffer, GL_UNSIGNED_BYTE, width, height)
    return ret != 0

def OSMesaGetCurrentContext():
    if False:
        while True:
            i = 10
    return c_void_p(_lib.OSMesaGetCurrentContext())
if __name__ == '__main__':
    'This test basic OSMesa functionality'
    context = OSMesaCreateContext()
    (w, h) = (640, 480)
    pixels = allocate_pixels_buffer(w, h)
    ok = OSMesaMakeCurrent(context, pixels, 640, 480)
    if not ok:
        raise RuntimeError('Failed OSMesaMakeCurrent')
    if not OSMesaGetCurrentContext().value == context.value:
        raise RuntimeError('OSMesa context not correctly attached')
    _lib.glGetString.argtypes = (ctypes.c_uint,)
    _lib.glGetString.restype = ctypes.c_char_p
    print('OpenGL version : ', _lib.glGetString(GL_VERSION))
    OSMesaDestroyContext(context)
    if OSMesaGetCurrentContext().value is not None:
        raise RuntimeError('Failed to destroy OSMesa context')