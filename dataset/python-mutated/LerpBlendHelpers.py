"""LerpBlendHelpers module: contains LerpBlendHelpers class"""
__all__ = ['getBlend']
from panda3d.direct import EaseInBlendType, EaseInOutBlendType, EaseOutBlendType, NoBlendType
easeIn = EaseInBlendType()
easeOut = EaseOutBlendType()
easeInOut = EaseInOutBlendType()
noBlend = NoBlendType()

def getBlend(blendType):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the C++ blend class corresponding to blendType string\n    '
    if blendType == 'easeIn':
        return easeIn
    elif blendType == 'easeOut':
        return easeOut
    elif blendType == 'easeInOut':
        return easeInOut
    elif blendType == 'noBlend':
        return noBlend
    else:
        raise Exception('Error: LerpInterval.__getBlend: Unknown blend type')