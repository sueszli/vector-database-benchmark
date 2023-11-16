"""SoundInterval module: contains the SoundInterval class"""
__all__ = ['SoundInterval']
from panda3d.direct import CInterval
from direct.directnotify.DirectNotifyGlobal import directNotify
from . import Interval
import random

class SoundInterval(Interval.Interval):
    soundNum = 1
    notify = directNotify.newCategory('SoundInterval')

    def __init__(self, sound, loop=0, duration=0.0, name=None, volume=1.0, startTime=0.0, node=None, seamlessLoop=True, listenerNode=None, cutOff=None):
        if False:
            print('Hello World!')
        '__init__(sound, loop, name)\n        '
        id = 'Sound-%d' % SoundInterval.soundNum
        SoundInterval.soundNum += 1
        self.sound = sound
        if sound:
            self.soundDuration = sound.length()
        else:
            self.soundDuration = 0
        self.fLoop = loop
        self.volume = volume
        self.startTime = startTime
        self.node = node
        self.listenerNode = listenerNode
        self.cutOff = cutOff
        self._seamlessLoop = seamlessLoop
        if self._seamlessLoop:
            self._fLoop = True
        self._soundPlaying = False
        self._reverse = False
        if float(duration) == 0.0 and self.sound is not None:
            duration = max(self.soundDuration - self.startTime, 0)
        if name is None:
            name = id
        Interval.Interval.__init__(self, name, duration)

    def privInitialize(self, t):
        if False:
            while True:
                i = 10
        self._reverse = False
        t1 = t + self.startTime
        if t1 < 0.1:
            t1 = 0.0
        if t1 < self.soundDuration and (not (self._seamlessLoop and self._soundPlaying)):
            base.sfxPlayer.playSfx(self.sound, self.fLoop, 1, self.volume, t1, self.node, listenerNode=self.listenerNode, cutoff=self.cutOff)
            self._soundPlaying = True
        self.state = CInterval.SStarted
        self.currT = t

    def privInstant(self):
        if False:
            i = 10
            return i + 15
        pass

    def privStep(self, t):
        if False:
            while True:
                i = 10
        if self.state == CInterval.SPaused:
            t1 = t + self.startTime
            if t1 < self.soundDuration:
                base.sfxPlayer.playSfx(self.sound, self.fLoop, 1, self.volume, t1, self.node, listenerNode=self.listenerNode)
        if self.listenerNode and (not self.listenerNode.isEmpty()) and self.node and (not self.node.isEmpty()):
            base.sfxPlayer.setFinalVolume(self.sound, self.node, self.volume, self.listenerNode, self.cutOff)
        self.state = CInterval.SStarted
        self.currT = t

    def finish(self, *args, **kArgs):
        if False:
            return 10
        self._inFinish = True
        Interval.Interval.finish(self, *args, **kArgs)
        del self._inFinish

    def privFinalize(self):
        if False:
            for i in range(10):
                print('nop')
        if self._seamlessLoop and self._soundPlaying and self.getLoop() and (not hasattr(self, '_inFinish')):
            base.sfxPlayer.setFinalVolume(self.sound, self.node, self.volume, self.listenerNode, self.cutOff)
            return
        elif self.sound is not None:
            self.sound.stop()
            self._soundPlaying = False
        self.currT = self.getDuration()
        self.state = CInterval.SFinal

    def privReverseInitialize(self, t):
        if False:
            while True:
                i = 10
        self._reverse = True

    def privReverseInstant(self):
        if False:
            return 10
        self.state = CInterval.SInitial

    def privReverseFinalize(self):
        if False:
            for i in range(10):
                print('nop')
        self._reverse = False
        self.state = CInterval.SInitial

    def privInterrupt(self):
        if False:
            i = 10
            return i + 15
        if self.sound is not None:
            self.sound.stop()
            self._soundPlaying = False
        self.state = CInterval.SPaused

    def loop(self, startT=0.0, endT=-1.0, playRate=1.0, stagger=False):
        if False:
            return 10
        self.fLoop = 1
        Interval.Interval.loop(self, startT, endT, playRate)
        if stagger:
            self.setT(random.random() * self.getDuration())