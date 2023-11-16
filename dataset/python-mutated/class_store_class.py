try:
    from collections import namedtuple
except ImportError:
    print('SKIP')
    raise SystemExit
_DefragResultBase = namedtuple('DefragResult', ['foo', 'bar'])

class _ResultMixinStr(object):

    def encode(self):
        if False:
            while True:
                i = 10
        return self._encoded_counterpart(*(x.encode() for x in self))

class _ResultMixinBytes(object):

    def decode(self):
        if False:
            print('Hello World!')
        return self._decoded_counterpart(*(x.decode() for x in self))

class DefragResult(_DefragResultBase, _ResultMixinStr):
    pass

class DefragResultBytes(_DefragResultBase, _ResultMixinBytes):
    pass
DefragResult._encoded_counterpart = DefragResultBytes
DefragResultBytes._decoded_counterpart = DefragResult
o1 = DefragResult('a', 'b')
o2 = DefragResultBytes('a', 'b')
_o1 = o1.encode()
print(_o1[0], _o1[1])
print("All's ok")