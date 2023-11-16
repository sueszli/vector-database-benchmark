from panda3d.core import ConfigVariableInt
from .DistributedObjectUD import DistributedObjectUD
from direct.directnotify.DirectNotifyGlobal import directNotify
import sys

class DistributedObjectGlobalUD(DistributedObjectUD):
    notify = directNotify.newCategory('DistributedObjectGlobalUD')
    doNotDeallocateChannel = 1
    isGlobalDistObj = 1

    def __init__(self, air):
        if False:
            i = 10
            return i + 15
        DistributedObjectUD.__init__(self, air)
        self.ExecNamespace = {'self': self}

    def announceGenerate(self):
        if False:
            return 10
        self.air.registerForChannel(self.doId)
        DistributedObjectUD.announceGenerate(self)

    def delete(self):
        if False:
            while True:
                i = 10
        self.air.unregisterForChannel(self.doId)
        DistributedObjectUD.delete(self)

    def execCommand(self, command, mwMgrId, avId, zoneId):
        if False:
            return 10
        length = ConfigVariableInt('ai-debug-length', 300)
        text = str(self.__execMessage(command))[:length.value]
        self.notify.info(text)

    def __execMessage(self, message):
        if False:
            return 10
        if not self.ExecNamespace:
            import panda3d.core
            for (key, value) in panda3d.core.__dict__.items():
                if not key.startswith('__'):
                    self.ExecNamespace[key] = value
        try:
            return str(eval(message, globals(), self.ExecNamespace))
        except SyntaxError:
            try:
                exec(message, globals(), self.ExecNamespace)
                return 'ok'
            except:
                exception = sys.exc_info()[0]
                extraInfo = sys.exc_info()[1]
                if extraInfo:
                    return str(extraInfo)
                else:
                    return str(exception)
        except:
            exception = sys.exc_info()[0]
            extraInfo = sys.exc_info()[1]
            if extraInfo:
                return str(extraInfo)
            else:
                return str(exception)