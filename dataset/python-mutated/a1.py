from .pyi_testmod_relimp3c import c1
from ..pyi_testmod_relimp3b import b1

def getString():
    if False:
        i = 10
        return i + 15
    return b1.string + c1.string