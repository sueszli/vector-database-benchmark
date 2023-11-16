"""MopathInterval module: contains the MopathInterval class"""
__all__ = ['MopathInterval']
from . import LerpInterval
from direct.directnotify.DirectNotifyGlobal import directNotify

class MopathInterval(LerpInterval.LerpFunctionInterval):
    mopathNum = 1
    notify = directNotify.newCategory('MopathInterval')

    def __init__(self, mopath, node, fromT=0, toT=None, duration=None, blendType='noBlend', name=None):
        if False:
            while True:
                i = 10
        if toT is None:
            toT = mopath.getMaxT()
        if duration is None:
            duration = abs(toT - fromT)
        if name is None:
            name = 'Mopath-%d' % MopathInterval.mopathNum
            MopathInterval.mopathNum += 1
        LerpInterval.LerpFunctionInterval.__init__(self, self.__doMopath, fromData=fromT, toData=toT, duration=duration, blendType=blendType, name=name)
        self.mopath = mopath
        self.node = node

    def destroy(self):
        if False:
            return 10
        'Cleanup to avoid a garbage cycle.'
        self.function = None

    def __doMopath(self, t):
        if False:
            return 10
        '\n        Go to time t\n        '
        self.mopath.goTo(self.node, t)