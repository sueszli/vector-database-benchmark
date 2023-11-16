"""Contains the SfxPlayer class, a thin utility class for playing sounds at
a particular location."""
__all__ = ['SfxPlayer']
import math
from panda3d.core import AudioSound

class SfxPlayer:
    """
    Play sound effects, potentially localized.
    """
    UseInverseSquare = 0

    def __init__(self):
        if False:
            while True:
                i = 10
        self.cutoffVolume = 0.02
        if SfxPlayer.UseInverseSquare:
            self.setCutoffDistance(300.0)
        else:
            self.setCutoffDistance(120.0)

    def setCutoffDistance(self, d):
        if False:
            while True:
                i = 10
        self.cutoffDistance = d
        rawCutoffDistance = math.sqrt(1.0 / self.cutoffVolume)
        self.distanceScale = rawCutoffDistance / self.cutoffDistance

    def getCutoffDistance(self):
        if False:
            i = 10
            return i + 15
        'Return the curent cutoff distance.'
        return self.cutoffDistance

    def getLocalizedVolume(self, node, listenerNode=None, cutoff=None):
        if False:
            return 10
        '\n        Get the volume that a sound should be played at if it is\n        localized at this node. We compute this wrt the camera\n        or to listenerNode.\n        '
        d = None
        if not node.isEmpty():
            if listenerNode and (not listenerNode.isEmpty()):
                d = node.getDistance(listenerNode)
            else:
                d = node.getDistance(base.cam)
        if not cutoff:
            cutoff = self.cutoffDistance
        if d is None or d > cutoff:
            volume = 0
        elif SfxPlayer.UseInverseSquare:
            sd = d * self.distanceScale
            volume = min(1, 1 / (sd * sd or 1))
        else:
            volume = 1 - d / (cutoff or 1)
        return volume

    def playSfx(self, sfx, looping=0, interrupt=1, volume=None, time=0.0, node=None, listenerNode=None, cutoff=None):
        if False:
            while True:
                i = 10
        if sfx:
            self.setFinalVolume(sfx, node, volume, listenerNode, cutoff)
            if interrupt or sfx.status() != AudioSound.PLAYING:
                sfx.setTime(time)
                sfx.setLoop(looping)
                sfx.play()

    def setFinalVolume(self, sfx, node, volume, listenerNode, cutoff=None):
        if False:
            print('Hello World!')
        'Calculate the final volume based on all contributed factors.'
        if node or volume is not None:
            if node:
                finalVolume = self.getLocalizedVolume(node, listenerNode, cutoff)
            else:
                finalVolume = 1
            if volume is not None:
                finalVolume *= volume
            if node is not None:
                finalVolume *= node.getNetAudioVolume()
            sfx.setVolume(finalVolume)