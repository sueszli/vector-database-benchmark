"""LerpInterval module: contains the LerpInterval class"""
__all__ = ['LerpNodePathInterval', 'LerpPosInterval', 'LerpHprInterval', 'LerpQuatInterval', 'LerpScaleInterval', 'LerpShearInterval', 'LerpPosHprInterval', 'LerpPosQuatInterval', 'LerpHprScaleInterval', 'LerpQuatScaleInterval', 'LerpPosHprScaleInterval', 'LerpPosQuatScaleInterval', 'LerpPosHprScaleShearInterval', 'LerpPosQuatScaleShearInterval', 'LerpColorInterval', 'LerpColorScaleInterval', 'LerpTexOffsetInterval', 'LerpTexRotateInterval', 'LerpTexScaleInterval', 'LerpFunctionInterval', 'LerpFunc', 'LerpFunctionNoStateInterval', 'LerpFuncNS']
from panda3d.core import LOrientationf, NodePath
from panda3d.direct import CInterval, CLerpNodePathInterval
from direct.directnotify.DirectNotifyGlobal import directNotify
from . import Interval
from . import LerpBlendHelpers

class LerpNodePathInterval(CLerpNodePathInterval):
    lerpNodePathNum = 1

    def __init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other):
        if False:
            i = 10
            return i + 15
        if name is None:
            name = '%s-%d' % (self.__class__.__name__, self.lerpNodePathNum)
            LerpNodePathInterval.lerpNodePathNum += 1
        elif '%d' in name:
            name = name % LerpNodePathInterval.lerpNodePathNum
            LerpNodePathInterval.lerpNodePathNum += 1
        blendType = self.stringBlendType(blendType)
        assert blendType != self.BTInvalid
        if other is None:
            other = NodePath()
        CLerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)

    def anyCallable(self, *params):
        if False:
            i = 10
            return i + 15
        for param in params:
            if callable(param):
                return 1
        return 0

    def setupParam(self, func, param):
        if False:
            return 10
        if param is not None:
            if callable(param):
                func(param())
            else:
                func(param)

class LerpPosInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, startPos=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            i = 10
            return i + 15
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(pos, startPos)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)

    def privDoEvent(self, t, event):
        if False:
            print('Hello World!')
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpHprInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, hpr, startHpr=None, startQuat=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            while True:
                i = 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(hpr, startHpr, startQuat)
        if self.paramSetup:
            self.endHpr = hpr
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.inPython = 1
        else:
            self.setEndHpr(hpr)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)

    def privDoEvent(self, t, event):
        if False:
            i = 10
            return i + 15
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndHpr, self.endHpr)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpQuatInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, quat=None, startHpr=None, startQuat=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None, hpr=None):
        if False:
            for i in range(10):
                print('nop')
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        if not quat:
            assert hpr
            quat = LOrientationf()
            quat.setHpr(hpr)
        self.paramSetup = self.anyCallable(quat, startHpr, startQuat)
        if self.paramSetup:
            self.endQuat = quat
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.inPython = 1
        else:
            self.setEndQuat(quat)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)

    def privDoEvent(self, t, event):
        if False:
            print('Hello World!')
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndQuat, self.endQuat)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, scale, startScale=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            print('Hello World!')
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(scale, startScale)
        if self.paramSetup:
            self.endScale = scale
            self.startScale = startScale
            self.inPython = 1
        else:
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)

    def privDoEvent(self, t, event):
        if False:
            for i in range(10):
                print('nop')
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpShearInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, shear, startShear=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            print('Hello World!')
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(shear, startShear)
        if self.paramSetup:
            self.endShear = shear
            self.startShear = startShear
            self.inPython = 1
        else:
            self.setEndShear(shear)
            if startShear is not None:
                self.setStartShear(startShear)

    def privDoEvent(self, t, event):
        if False:
            i = 10
            return i + 15
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndShear, self.endShear)
            self.setupParam(self.setStartShear, self.startShear)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpPosHprInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, hpr, startPos=None, startHpr=None, startQuat=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            i = 10
            return i + 15
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(pos, startPos, hpr, startHpr, startQuat)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.endHpr = hpr
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)
            self.setEndHpr(hpr)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)

    def privDoEvent(self, t, event):
        if False:
            return 10
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
            self.setupParam(self.setEndHpr, self.endHpr)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpPosQuatInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, quat=None, startPos=None, startHpr=None, startQuat=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None, hpr=None):
        if False:
            while True:
                i = 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        if not quat:
            assert hpr
            quat = LOrientationf()
            quat.setHpr(hpr)
        self.paramSetup = self.anyCallable(pos, startPos, quat, startHpr, startQuat)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.endQuat = quat
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)
            self.setEndQuat(quat)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)

    def privDoEvent(self, t, event):
        if False:
            i = 10
            return i + 15
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
            self.setupParam(self.setEndQuat, self.endQuat)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpHprScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, hpr, scale, startHpr=None, startQuat=None, startScale=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            while True:
                i = 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(hpr, startHpr, startQuat, scale, startScale)
        if self.paramSetup:
            self.endHpr = hpr
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.endScale = scale
            self.startScale = startScale
            self.inPython = 1
        else:
            self.setEndHpr(hpr)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)

    def privDoEvent(self, t, event):
        if False:
            while True:
                i = 10
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndHpr, self.endHpr)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpQuatScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, quat=None, scale=None, hpr=None, startHpr=None, startQuat=None, startScale=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            return 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        if not quat:
            assert hpr
            quat = LOrientationf()
            quat.setHpr(hpr)
        assert scale
        self.paramSetup = self.anyCallable(quat, startHpr, startQuat, scale, startScale)
        if self.paramSetup:
            self.endQuat = quat
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.endScale = scale
            self.startScale = startScale
            self.inPython = 1
        else:
            self.setEndQuat(quat)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)

    def privDoEvent(self, t, event):
        if False:
            i = 10
            return i + 15
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndQuat, self.endQuat)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpPosHprScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, hpr, scale, startPos=None, startHpr=None, startQuat=None, startScale=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            i = 10
            return i + 15
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(pos, startPos, hpr, startHpr, startQuat, scale, startScale)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.endHpr = hpr
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.endScale = scale
            self.startScale = startScale
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)
            self.setEndHpr(hpr)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)

    def privDoEvent(self, t, event):
        if False:
            for i in range(10):
                print('nop')
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
            self.setupParam(self.setEndHpr, self.endHpr)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpPosQuatScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, quat=None, scale=None, startPos=None, startHpr=None, startQuat=None, startScale=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None, hpr=None):
        if False:
            return 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        if not quat:
            assert hpr
            quat = LOrientationf()
            quat.setHpr(hpr)
        assert scale
        self.paramSetup = self.anyCallable(pos, startPos, quat, startHpr, startQuat, scale, startScale)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.endQuat = quat
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.endScale = scale
            self.startScale = startScale
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)
            self.setEndQuat(quat)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)

    def privDoEvent(self, t, event):
        if False:
            for i in range(10):
                print('nop')
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
            self.setupParam(self.setEndQuat, self.endQuat)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpPosHprScaleShearInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, hpr, scale, shear, startPos=None, startHpr=None, startQuat=None, startScale=None, startShear=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None):
        if False:
            i = 10
            return i + 15
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        self.paramSetup = self.anyCallable(pos, startPos, hpr, startHpr, startQuat, scale, startScale, shear, startShear)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.endHpr = hpr
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.endScale = scale
            self.startScale = startScale
            self.endShear = shear
            self.startShear = startShear
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)
            self.setEndHpr(hpr)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)
            self.setEndShear(shear)
            if startShear is not None:
                self.setStartShear(startShear)

    def privDoEvent(self, t, event):
        if False:
            for i in range(10):
                print('nop')
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
            self.setupParam(self.setEndHpr, self.endHpr)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
            self.setupParam(self.setEndShear, self.endShear)
            self.setupParam(self.setStartShear, self.startShear)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpPosQuatScaleShearInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, pos, quat=None, scale=None, shear=None, startPos=None, startHpr=None, startQuat=None, startScale=None, startShear=None, other=None, blendType='noBlend', bakeInStart=1, fluid=0, name=None, hpr=None):
        if False:
            return 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, fluid, nodePath, other)
        if not quat:
            assert hpr
            quat = LOrientationf()
            quat.setHpr(hpr)
        assert scale
        assert shear
        self.paramSetup = self.anyCallable(pos, startPos, quat, startHpr, startQuat, scale, startScale, shear, startShear)
        if self.paramSetup:
            self.endPos = pos
            self.startPos = startPos
            self.endQuat = quat
            self.startHpr = startHpr
            self.startQuat = startQuat
            self.endScale = scale
            self.startScale = startScale
            self.endShear = shear
            self.startShear = startShear
            self.inPython = 1
        else:
            self.setEndPos(pos)
            if startPos is not None:
                self.setStartPos(startPos)
            self.setEndQuat(quat)
            if startHpr is not None:
                self.setStartHpr(startHpr)
            if startQuat is not None:
                self.setStartQuat(startQuat)
            self.setEndScale(scale)
            if startScale is not None:
                self.setStartScale(startScale)
            self.setEndShear(shear)
            if startShear is not None:
                self.setStartShear(startShear)

    def privDoEvent(self, t, event):
        if False:
            while True:
                i = 10
        if self.paramSetup and event == CInterval.ETInitialize:
            self.setupParam(self.setEndPos, self.endPos)
            self.setupParam(self.setStartPos, self.startPos)
            self.setupParam(self.setEndQuat, self.endQuat)
            self.setupParam(self.setStartHpr, self.startHpr)
            self.setupParam(self.setStartQuat, self.startQuat)
            self.setupParam(self.setEndScale, self.endScale)
            self.setupParam(self.setStartScale, self.startScale)
            self.setupParam(self.setEndShear, self.endShear)
            self.setupParam(self.setStartShear, self.startShear)
        LerpNodePathInterval.privDoEvent(self, t, event)

class LerpColorInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, color, startColor=None, other=None, blendType='noBlend', bakeInStart=1, name=None, override=None):
        if False:
            return 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, 0, nodePath, other)
        self.setEndColor(color)
        if startColor is not None:
            self.setStartColor(startColor)
        if override is not None:
            self.setOverride(override)

class LerpColorScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, colorScale, startColorScale=None, other=None, blendType='noBlend', bakeInStart=1, name=None, override=None):
        if False:
            while True:
                i = 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, 0, nodePath, other)
        self.setEndColorScale(colorScale)
        if startColorScale is not None:
            self.setStartColorScale(startColorScale)
        if override is not None:
            self.setOverride(override)

class LerpTexOffsetInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, texOffset, startTexOffset=None, other=None, blendType='noBlend', textureStage=None, bakeInStart=1, name=None, override=None):
        if False:
            return 10
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, 0, nodePath, other)
        self.setEndTexOffset(texOffset)
        if startTexOffset is not None:
            self.setStartTexOffset(startTexOffset)
        if textureStage is not None:
            self.setTextureStage(textureStage)
        if override is not None:
            self.setOverride(override)

class LerpTexRotateInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, texRotate, startTexRotate=None, other=None, blendType='noBlend', textureStage=None, bakeInStart=1, name=None, override=None):
        if False:
            i = 10
            return i + 15
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, 0, nodePath, other)
        self.setEndTexRotate(texRotate)
        if startTexRotate is not None:
            self.setStartTexRotate(startTexRotate)
        if textureStage is not None:
            self.setTextureStage(textureStage)
        if override is not None:
            self.setOverride(override)

class LerpTexScaleInterval(LerpNodePathInterval):

    def __init__(self, nodePath, duration, texScale, startTexScale=None, other=None, blendType='noBlend', textureStage=None, bakeInStart=1, name=None, override=None):
        if False:
            for i in range(10):
                print('nop')
        LerpNodePathInterval.__init__(self, name, duration, blendType, bakeInStart, 0, nodePath, other)
        self.setEndTexScale(texScale)
        if startTexScale is not None:
            self.setStartTexScale(startTexScale)
        if textureStage is not None:
            self.setTextureStage(textureStage)
        if override is not None:
            self.setOverride(override)

class LerpFunctionNoStateInterval(Interval.Interval):
    """
    Class used to execute a function over time.  Function can access fromData
    and toData to perform blend.  If fromData and toData not specified, will
    execute the given function passing in values ranging from 0 to 1

    This is different from a standard LerpFunction, in that it assumes
    the function is not modifying any state that needs to be kept; so
    that it will only call the function while the lerp is actually
    running, and will not be guaranteed to call the function with its
    final value of the lerp.  In particular, if the lerp interval
    happens to get skipped over completely, it will not bother to call
    the function at all.
    """
    lerpFunctionIntervalNum = 1
    notify = directNotify.newCategory('LerpFunctionNoStateInterval')

    def __init__(self, function, duration=0.0, fromData=0, toData=1, blendType='noBlend', extraArgs=[], name=None):
        if False:
            for i in range(10):
                print('nop')
        '__init__(function, duration, fromData, toData, name)\n        '
        self.function = function
        self.fromData = fromData
        self.toData = toData
        self.blendType = LerpBlendHelpers.getBlend(blendType)
        self.extraArgs = extraArgs
        if name is None:
            name = 'LerpFunctionInterval-%d' % LerpFunctionNoStateInterval.lerpFunctionIntervalNum
            LerpFunctionNoStateInterval.lerpFunctionIntervalNum += 1
        elif '%d' in name:
            name = name % LerpFunctionNoStateInterval.lerpFunctionIntervalNum
            LerpFunctionNoStateInterval.lerpFunctionIntervalNum += 1
        Interval.Interval.__init__(self, name, duration)

    def privStep(self, t):
        if False:
            while True:
                i = 10
        if t >= self.duration:
            if t > self.duration:
                print('after end')
        elif self.duration == 0.0:
            self.function(*[self.toData] + self.extraArgs)
        else:
            bt = self.blendType(t / self.duration)
            data = self.fromData * (1 - bt) + self.toData * bt
            self.function(*[data] + self.extraArgs)
        self.state = CInterval.SStarted
        self.currT = t

class LerpFuncNS(LerpFunctionNoStateInterval):

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        LerpFunctionNoStateInterval.__init__(self, *args, **kw)

class LerpFunctionInterval(Interval.Interval):
    """
    Class used to execute a function over time.  Function can access fromData
    and toData to perform blend.  If fromData and toData not specified, will
    execute the given function passing in values ranging from 0 to 1
    """
    lerpFunctionIntervalNum = 1
    notify = directNotify.newCategory('LerpFunctionInterval')

    def __init__(self, function, duration=0.0, fromData=0, toData=1, blendType='noBlend', extraArgs=[], name=None):
        if False:
            print('Hello World!')
        '__init__(function, duration, fromData, toData, name)\n        '
        self.function = function
        self.fromData = fromData
        self.toData = toData
        self.blendType = LerpBlendHelpers.getBlend(blendType)
        self.extraArgs = extraArgs
        if name is None:
            name = 'LerpFunctionInterval-%s-%d' % (function.__name__, LerpFunctionInterval.lerpFunctionIntervalNum)
            LerpFunctionInterval.lerpFunctionIntervalNum += 1
        elif '%d' in name:
            name = name % LerpFunctionInterval.lerpFunctionIntervalNum
            LerpFunctionInterval.lerpFunctionIntervalNum += 1
        Interval.Interval.__init__(self, name, duration)

    def privStep(self, t):
        if False:
            while True:
                i = 10
        if t >= self.duration:
            self.function(*[self.toData] + self.extraArgs)
        elif self.duration == 0.0:
            self.function(*[self.toData] + self.extraArgs)
        else:
            bt = self.blendType(t / self.duration)
            data = self.fromData * (1 - bt) + self.toData * bt
            self.function(*[data] + self.extraArgs)
        self.state = CInterval.SStarted
        self.currT = t

class LerpFunc(LerpFunctionInterval):

    def __init__(self, *args, **kw):
        if False:
            return 10
        LerpFunctionInterval.__init__(self, *args, **kw)