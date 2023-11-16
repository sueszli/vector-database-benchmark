import Ogre
import ctypes
import numpy.ctypeslib as npc
import numpy as np

def AsDataStream(arr):
    if False:
        while True:
            i = 10
    '\n    copy numpy array to Ogre.MemoryDataStream that can be used in Ogre\n    @param arr: some numpy array\n    '
    size = int(np.prod(arr.shape) * arr.dtype.itemsize)
    ret = Ogre.MemoryDataStream(size)
    tp = ctypes.POINTER(ctypes.c_ubyte)
    np_view = npc.as_array(ctypes.cast(int(ret.getPtr()), tp), (size,))
    np_view[:] = arr.ravel().view(np.ubyte)
    return ret

def view(o):
    if False:
        i = 10
        return i + 15
    '\n    writable numpy view to the ogre data types\n    \n    take care that the ogre type does not get released while the view is used.\n    e.g. this is invalid\n    \n    v = Ogre.Vector3()\n    return OgreNumpy.view(v)\n    \n    instead do\n    return OgreNumpy.view(v).copy()\n    \n    to pass numpy arrays into Ogre use AsDataStream()\n    '
    tp = ctypes.POINTER(ctypes.c_float)
    if isinstance(o, Ogre.Vector2):
        shape = (2,)
        ptr = o.this
    elif isinstance(o, Ogre.Vector3):
        shape = (3,)
        ptr = o.this
    elif isinstance(o, Ogre.Vector4):
        shape = (4,)
        ptr = o.this
    elif isinstance(o, Ogre.Matrix3):
        shape = (3, 3)
        ptr = o.this
    elif isinstance(o, Ogre.Matrix4):
        shape = (4, 4)
        ptr = o.this
    elif isinstance(o, Ogre.PixelBox):
        tp = ctypes.POINTER(ctypes.c_uint8)
        shape = (o.getHeight(), o.getWidth(), Ogre.PixelUtil.getNumElemBytes(o.format))
        ptr = o.data
    else:
        raise TypeError("do not know how to map '{}'".format(type(o).__name__))
    return npc.as_array(ctypes.cast(int(ptr), tp), shape)