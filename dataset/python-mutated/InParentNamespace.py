import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class InParentNamespace(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = InParentNamespace()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsInParentNamespace(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def InParentNamespaceBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            print('Hello World!')
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

def InParentNamespaceStart(builder):
    if False:
        for i in range(10):
            print('nop')
    builder.StartObject(0)

def Start(builder):
    if False:
        print('Hello World!')
    InParentNamespaceStart(builder)

def InParentNamespaceEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        return 10
    return InParentNamespaceEnd(builder)

class InParentNamespaceT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            i = 10
            return i + 15
        inParentNamespace = InParentNamespace()
        inParentNamespace.Init(buf, pos)
        return cls.InitFromObj(inParentNamespace)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, inParentNamespace):
        if False:
            i = 10
            return i + 15
        x = InParentNamespaceT()
        x._UnPack(inParentNamespace)
        return x

    def _UnPack(self, inParentNamespace):
        if False:
            print('Hello World!')
        if inParentNamespace is None:
            return

    def Pack(self, builder):
        if False:
            print('Hello World!')
        InParentNamespaceStart(builder)
        inParentNamespace = InParentNamespaceEnd(builder)
        return inParentNamespace