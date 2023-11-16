from twisted.internet import reactor
from twisted.spread import pb

class MyError(pb.Error):
    """This is an Expected Exception. Something bad happened."""
    pass

class MyError2(Exception):
    """This is an Unexpected Exception. Something really bad happened."""
    pass

class One(pb.Root):

    def remote_broken(self):
        if False:
            print('Hello World!')
        msg = 'fall down go boom'
        print("raising a MyError exception with data '%s'" % msg)
        raise MyError(msg)

    def remote_broken2(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'hadda owie'
        print("raising a MyError2 exception with data '%s'" % msg)
        raise MyError2(msg)

def main():
    if False:
        return 10
    reactor.listenTCP(8800, pb.PBServerFactory(One()))
    reactor.run()
if __name__ == '__main__':
    main()