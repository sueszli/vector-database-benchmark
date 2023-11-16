""" Showutil Effects module: contains code for useful showcode effects. """
from panda3d.core import Vec3
from direct.interval.IntervalGlobal import LerpHprInterval, LerpPosInterval, LerpScaleInterval, Sequence
SX_BOUNCE = 0
SY_BOUNCE = 1
SZ_BOUNCE = 2
TX_BOUNCE = 3
TY_BOUNCE = 4
TZ_BOUNCE = 5
H_BOUNCE = 6
P_BOUNCE = 7
R_BOUNCE = 8

def createScaleXBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        return 10
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, SX_BOUNCE)

def createScaleYBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        for i in range(10):
            print('nop')
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, SY_BOUNCE)

def createScaleZBounce(nodeObj, numBounces, startValue, totalTime, amplitude):
    if False:
        for i in range(10):
            print('nop')
    return createBounce(nodeObj, numBounces, startValue, totalTime, amplitude, SZ_BOUNCE)

def createXBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        print('Hello World!')
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, TX_BOUNCE)

def createYBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        while True:
            i = 10
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, TY_BOUNCE)

def createZBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        return 10
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, TZ_BOUNCE)

def createHBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        return 10
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, H_BOUNCE)

def createPBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        for i in range(10):
            print('nop')
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, P_BOUNCE)

def createRBounce(nodeObj, numBounces, startValues, totalTime, amplitude):
    if False:
        i = 10
        return i + 15
    return createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, R_BOUNCE)

def createBounce(nodeObj, numBounces, startValues, totalTime, amplitude, bounceType=SZ_BOUNCE):
    if False:
        i = 10
        return i + 15
    '\n    createBounce: create and return a list of intervals to make a\n    given nodePath bounce a given number of times over a give total time.\n    '
    if not nodeObj or numBounces < 1 or totalTime == 0:
        raise ValueError('createBounce called with invalid parameter')
    result = Sequence()
    bounceTime = totalTime / numBounces
    currTime = bounceTime
    currAmplitude = amplitude
    index = bounceType % 3
    currBounceVal = startValues[index]
    for bounceNum in range(numBounces * 2):
        if bounceNum % 2:
            currBounceVal = startValues[index]
            blend = 'easeIn'
        else:
            currBounceVal = startValues[index] + currAmplitude
            blend = 'easeOut'
        newVec3 = Vec3(startValues)
        newVec3.setCell(index, currBounceVal)
        if bounceType >= SX_BOUNCE and bounceType <= SZ_BOUNCE:
            result.append(LerpScaleInterval(nodeObj, currTime, newVec3, blendType=blend))
        elif bounceType >= TX_BOUNCE and bounceType <= TZ_BOUNCE:
            result.append(LerpPosInterval(nodeObj, currTime, newVec3, blendType=blend))
        elif bounceType >= H_BOUNCE and bounceType <= R_BOUNCE:
            result.append(LerpHprInterval(nodeObj, currTime, newVec3, blendType=blend))
        currAmplitude *= 0.5
        currTime = bounceTime
    return result