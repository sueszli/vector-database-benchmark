"""AnimControlInterval module: contains the AnimControlInterval class"""
__all__ = ['AnimControlInterval']
from panda3d.core import AnimControl, AnimControlCollection, ConfigVariableBool
from panda3d.direct import CInterval
from direct.directnotify.DirectNotifyGlobal import directNotify
from . import Interval
import math

class AnimControlInterval(Interval.Interval):
    notify = directNotify.newCategory('AnimControlInterval')
    animNum = 1

    def __init__(self, controls, loop=0, constrainedLoop=0, duration=None, startTime=None, endTime=None, startFrame=None, endFrame=None, playRate=1.0, name=None):
        if False:
            i = 10
            return i + 15
        id = 'AnimControl-%d' % AnimControlInterval.animNum
        AnimControlInterval.animNum += 1
        if isinstance(controls, AnimControlCollection):
            self.controls = controls
            if ConfigVariableBool('strict-anim-ival', 0):
                checkSz = self.controls.getAnim(0).getNumFrames()
                for i in range(1, self.controls.getNumAnims()):
                    if checkSz != self.controls.getAnim(i).getNumFrames():
                        self.notify.error("anim controls don't have the same number of frames!")
        elif isinstance(controls, AnimControl):
            self.controls = AnimControlCollection()
            self.controls.storeAnim(controls, '')
        else:
            self.notify.error('invalid input control(s) for AnimControlInterval')
        self.loopAnim = loop
        self.constrainedLoop = constrainedLoop
        self.playRate = playRate
        if name is None:
            name = id
        self.frameRate = self.controls.getAnim(0).getFrameRate() * abs(playRate)
        if startFrame is not None:
            self.startFrame = startFrame
        elif startTime is not None:
            self.startFrame = startTime * self.frameRate
        else:
            self.startFrame = 0
        if endFrame is not None:
            self.endFrame = endFrame
        elif endTime is not None:
            self.endFrame = endTime * self.frameRate
        elif duration is not None:
            if startTime is None:
                startTime = float(self.startFrame) / float(self.frameRate)
            endTime = startTime + duration
            self.endFrame = duration * self.frameRate
        else:
            numFrames = self.controls.getAnim(0).getNumFrames()
            self.endFrame = numFrames - 1
        self.reverse = playRate < 0
        if self.endFrame < self.startFrame:
            self.reverse = 1
            t = self.endFrame
            self.endFrame = self.startFrame
            self.startFrame = t
        self.numFrames = self.endFrame - self.startFrame + 1
        self.implicitDuration = 0
        if duration is None:
            self.implicitDuration = 1
            duration = float(self.numFrames) / self.frameRate
        Interval.Interval.__init__(self, name, duration)

    def getCurrentFrame(self):
        if False:
            print('Hello World!')
        'Calculate the current frame playing in this interval.\n\n        returns a float value between startFrame and endFrame, inclusive\n        returns None if there are any problems\n        '
        retval = None
        if not self.isStopped():
            framesPlayed = self.numFrames * self.currT
            retval = self.startFrame + framesPlayed
        return retval

    def privStep(self, t):
        if False:
            while True:
                i = 10
        frameCount = t * self.frameRate
        if self.constrainedLoop:
            frameCount = frameCount % self.numFrames
        if self.reverse:
            absFrame = self.endFrame - frameCount
        else:
            absFrame = self.startFrame + frameCount
        intFrame = int(math.floor(absFrame + 0.0001))
        numFrames = self.controls.getAnim(0).getNumFrames()
        if self.loopAnim:
            frame = intFrame % numFrames + (absFrame - intFrame)
        else:
            frame = max(min(absFrame, numFrames - 1), 0)
        self.controls.poseAll(frame)
        self.state = CInterval.SStarted
        self.currT = t

    def privFinalize(self):
        if False:
            for i in range(10):
                print('nop')
        if self.implicitDuration and (not self.loopAnim):
            if self.reverse:
                self.controls.poseAll(self.startFrame)
            else:
                self.controls.poseAll(self.endFrame)
        else:
            self.privStep(self.getDuration())
        self.state = CInterval.SFinal
        self.intervalDone()