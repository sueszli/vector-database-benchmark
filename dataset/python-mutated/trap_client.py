from twisted.internet import reactor
from twisted.python import log
from twisted.spread import jelly, pb

class MyException(pb.Error):
    pass

class MyOtherException(pb.Error):
    pass

class ScaryObject:
    pass

def worksLike(obj):
    if False:
        while True:
            i = 10
    try:
        response = obj.callMethod(name, arg)
    except pb.DeadReferenceError:
        print(' stale reference: the client disconnected or crashed')
    except jelly.InsecureJelly:
        print(' InsecureJelly: you tried to send something unsafe to them')
    except (MyException, MyOtherException):
        print(' remote raised a MyException')
    except BaseException:
        print(' something else happened')
    else:
        print(' method successful, response:', response)

class One:

    def worked(self, response):
        if False:
            while True:
                i = 10
        print(' method successful, response:', response)

    def check_InsecureJelly(self, failure):
        if False:
            i = 10
            return i + 15
        failure.trap(jelly.InsecureJelly)
        print(' InsecureJelly: you tried to send something unsafe to them')
        return None

    def check_MyException(self, failure):
        if False:
            for i in range(10):
                print('nop')
        which = failure.trap(MyException, MyOtherException)
        if which == MyException:
            print(' remote raised a MyException')
        else:
            print(' remote raised a MyOtherException')
        return None

    def catch_everythingElse(self, failure):
        if False:
            return 10
        print(' something else happened')
        log.err(failure)
        return None

    def doCall(self, explanation, arg):
        if False:
            while True:
                i = 10
        print(explanation)
        try:
            deferred = self.remote.callRemote('fooMethod', arg)
            deferred.addCallback(self.worked)
            deferred.addErrback(self.check_InsecureJelly)
            deferred.addErrback(self.check_MyException)
            deferred.addErrback(self.catch_everythingElse)
        except pb.DeadReferenceError:
            print(' stale reference: the client disconnected or crashed')

    def callOne(self):
        if False:
            print('Hello World!')
        self.doCall('callOne: call with safe object', 'safe string')

    def callTwo(self):
        if False:
            print('Hello World!')
        self.doCall('callTwo: call with dangerous object', ScaryObject())

    def callThree(self):
        if False:
            i = 10
            return i + 15
        self.doCall('callThree: call that raises remote exception', 'panic!')

    def callShutdown(self):
        if False:
            print('Hello World!')
        print('telling them to shut down')
        self.remote.callRemote('shutdown')

    def callFour(self):
        if False:
            i = 10
            return i + 15
        self.doCall('callFour: call on stale reference', 'dummy')

    def got_obj(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self.remote = obj
        reactor.callLater(1, self.callOne)
        reactor.callLater(2, self.callTwo)
        reactor.callLater(3, self.callThree)
        reactor.callLater(4, self.callShutdown)
        reactor.callLater(5, self.callFour)
        reactor.callLater(6, reactor.stop)
factory = pb.PBClientFactory()
reactor.connectTCP('localhost', 8800, factory)
deferred = factory.getRootObject()
deferred.addCallback(One().got_obj)
reactor.run()