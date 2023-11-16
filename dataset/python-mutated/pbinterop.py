"""PB interop server."""
from twisted.internet import reactor
from twisted.spread import flavors, pb

class Interop(pb.Root):
    """Test object for PB interop tests."""

    def __init__(self):
        if False:
            return 10
        self.o = pb.Referenceable()

    def remote_int(self):
        if False:
            print('Hello World!')
        return 1

    def remote_string(self):
        if False:
            for i in range(10):
                print('nop')
        return 'string'

    def remote_unicode(self):
        if False:
            print('Hello World!')
        return 'string'

    def remote_float(self):
        if False:
            while True:
                i = 10
        return 1.5

    def remote_list(self):
        if False:
            i = 10
            return i + 15
        return [1, 2, 3]

    def remote_recursive(self):
        if False:
            for i in range(10):
                print('nop')
        l = []
        l.append(l)
        return l

    def remote_dict(self):
        if False:
            return 10
        return {1: 2}

    def remote_reference(self):
        if False:
            i = 10
            return i + 15
        return self.o

    def remote_local(self, obj):
        if False:
            i = 10
            return i + 15
        d = obj.callRemote('hello')
        d.addCallback(self._local_success)

    def _local_success(self, result):
        if False:
            print('Hello World!')
        if result != 'hello, world':
            raise ValueError('{} != {}'.format(result, 'hello, world'))

    def remote_receive(self, obj):
        if False:
            i = 10
            return i + 15
        expected = [1, 1.5, 'hi', 'hi', {1: 2}]
        if obj != expected:
            raise ValueError(f'{obj} != {expected}')

    def remote_self(self, obj):
        if False:
            print('Hello World!')
        if obj != self:
            raise ValueError(f'{obj} != {self}')

    def remote_copy(self, x):
        if False:
            i = 10
            return i + 15
        o = flavors.Copyable()
        o.x = x
        return o
if __name__ == '__main__':
    reactor.listenTCP(8789, pb.PBServerFactory(Interop()))
    reactor.run()