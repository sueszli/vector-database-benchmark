import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class HelloReply(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HelloReply()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsHelloReply(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Message(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def HelloReplyStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(1)

def Start(builder):
    if False:
        while True:
            i = 10
    HelloReplyStart(builder)

def HelloReplyAddMessage(builder, message):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(message), 0)

def AddMessage(builder, message):
    if False:
        print('Hello World!')
    HelloReplyAddMessage(builder, message)

def HelloReplyEnd(builder):
    if False:
        return 10
    return builder.EndObject()

def End(builder):
    if False:
        print('Hello World!')
    return HelloReplyEnd(builder)