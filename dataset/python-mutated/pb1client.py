from twisted.internet import reactor
from twisted.spread import pb

def main():
    if False:
        return 10
    factory = pb.PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    def1 = factory.getRootObject()
    def1.addCallbacks(got_obj1, err_obj1)
    reactor.run()

def err_obj1(reason):
    if False:
        i = 10
        return i + 15
    print('error getting first object', reason)
    reactor.stop()

def got_obj1(obj1):
    if False:
        for i in range(10):
            print('nop')
    print('got first object:', obj1)
    print('asking it to getTwo')
    def2 = obj1.callRemote('getTwo')
    def2.addCallbacks(got_obj2)

def got_obj2(obj2):
    if False:
        while True:
            i = 10
    print('got second object:', obj2)
    print('telling it to do three(12)')
    obj2.callRemote('three', 12)
main()