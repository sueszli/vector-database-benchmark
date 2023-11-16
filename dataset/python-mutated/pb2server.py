from twisted.internet import reactor
from twisted.spread import pb

class Two(pb.Referenceable):

    def remote_print(self, arg):
        if False:
            i = 10
            return i + 15
        print('two.print was given', arg)

class One(pb.Root):

    def __init__(self, two):
        if False:
            for i in range(10):
                print('nop')
        self.two = two

    def remote_getTwo(self):
        if False:
            print('Hello World!')
        print('One.getTwo(), returning my two called', self.two)
        return self.two

    def remote_checkTwo(self, newtwo):
        if False:
            print('Hello World!')
        print('One.checkTwo(): comparing my two', self.two)
        print('One.checkTwo(): against your two', newtwo)
        if self.two == newtwo:
            print('One.checkTwo(): our twos are the same')
two = Two()
root_obj = One(two)
reactor.listenTCP(8800, pb.PBServerFactory(root_obj))
reactor.run()