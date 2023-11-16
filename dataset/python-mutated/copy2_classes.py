from twisted.spread import pb

class FrogPond:

    def __init__(self, numFrogs, numToads):
        if False:
            for i in range(10):
                print('nop')
        self.numFrogs = numFrogs
        self.numToads = numToads

    def count(self):
        if False:
            return 10
        return self.numFrogs + self.numToads

class SenderPond(FrogPond, pb.Copyable):

    def getStateToCopy(self):
        if False:
            while True:
                i = 10
        d = self.__dict__.copy()
        d['frogsAndToads'] = d['numFrogs'] + d['numToads']
        del d['numFrogs']
        del d['numToads']
        return d

class ReceiverPond(pb.RemoteCopy):

    def setCopyableState(self, state):
        if False:
            return 10
        self.__dict__ = state

    def count(self):
        if False:
            for i in range(10):
                print('nop')
        return self.frogsAndToads
pb.setUnjellyableForClass(SenderPond, ReceiverPond)