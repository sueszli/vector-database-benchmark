"""SampleObject module: contains the SampleObject class"""
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.distributed.DistributedObject import DistributedObject

class SampleObject(DistributedObject):
    notify = directNotify.newCategory('SampleObject')

    def __init__(self, cr):
        if False:
            return 10
        self.cr = cr

    def setColor(self, red=0, green=0, blue=0):
        if False:
            for i in range(10):
                print('nop')
        self.red = red
        self.green = green
        self.blue = blue
        self.announceGenerate()

    def getColor(self):
        if False:
            while True:
                i = 10
        return (self.red, self.green, self.blue)