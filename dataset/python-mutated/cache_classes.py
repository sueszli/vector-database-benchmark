from twisted.spread import pb

class MasterDuckPond(pb.Cacheable):

    def __init__(self, ducks):
        if False:
            return 10
        self.observers = []
        self.ducks = ducks

    def count(self):
        if False:
            while True:
                i = 10
        print('I have [%d] ducks' % len(self.ducks))

    def addDuck(self, duck):
        if False:
            print('Hello World!')
        self.ducks.append(duck)
        for o in self.observers:
            o.callRemote('addDuck', duck)

    def removeDuck(self, duck):
        if False:
            return 10
        self.ducks.remove(duck)
        for o in self.observers:
            o.callRemote('removeDuck', duck)

    def getStateToCacheAndObserveFor(self, perspective, observer):
        if False:
            print('Hello World!')
        self.observers.append(observer)
        return self.ducks

    def stoppedObserving(self, perspective, observer):
        if False:
            print('Hello World!')
        self.observers.remove(observer)

class SlaveDuckPond(pb.RemoteCache):

    def count(self):
        if False:
            return 10
        return len(self.cacheducks)

    def getDucks(self):
        if False:
            return 10
        return self.cacheducks

    def setCopyableState(self, state):
        if False:
            for i in range(10):
                print('nop')
        print(' cache - sitting, er, setting ducks')
        self.cacheducks = state

    def observe_addDuck(self, newDuck):
        if False:
            return 10
        print(' cache - addDuck')
        self.cacheducks.append(newDuck)

    def observe_removeDuck(self, deadDuck):
        if False:
            for i in range(10):
                print('nop')
        print(' cache - removeDuck')
        self.cacheducks.remove(deadDuck)
pb.setUnjellyableForClass(MasterDuckPond, SlaveDuckPond)