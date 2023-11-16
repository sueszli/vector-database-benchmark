from direct.directnotify import DirectNotifyGlobal
from direct.distributed.PyDatagram import PyDatagram
from direct.showbase.Messenger import Messenger
from pickle import dumps, loads
MESSAGE_TYPES = ('avatarOnline', 'avatarOffline', 'create', 'needUberdogCreates', 'transferDo')
MESSAGE_STRINGS = {}
for i in zip(MESSAGE_TYPES, range(1, len(MESSAGE_TYPES) + 1)):
    MESSAGE_STRINGS[i[0]] = i[1]

class NetMessenger(Messenger):
    """
    This works very much like the Messenger class except that messages
    are sent over the network and (possibly) handled (accepted) on a
    remote machine (server).
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('NetMessenger')

    def __init__(self, air, channels):
        if False:
            while True:
                i = 10
        '\n        air is the AI Repository.\n        channels is a list of channel IDs (uint32 values)\n        '
        assert self.notify.debugCall()
        Messenger.__init__(self)
        self.air = air
        self.channels = channels
        for i in self.channels:
            self.air.registerForChannel(i)

    def clear(self):
        if False:
            print('Hello World!')
        assert self.notify.debugCall()
        for i in self.channels:
            self.air.unRegisterChannel(i)
        del self.air
        del self.channels
        Messenger.clear(self)

    def send(self, message, sentArgs=[]):
        if False:
            return 10
        '\n        Send message to All AI and Uber Dog servers.\n        '
        assert self.notify.debugCall()
        datagram = PyDatagram()
        datagram.addUint8(1)
        datagram.addChannel(self.channels[0])
        datagram.addChannel(self.air.ourChannel)
        messageType = MESSAGE_STRINGS.get(message, 0)
        datagram.addUint16(messageType)
        if messageType:
            datagram.addString(str(dumps(sentArgs)))
        else:
            datagram.addString(str(dumps((message, sentArgs))))
        self.air.send(datagram)

    def handle(self, pickleData):
        if False:
            i = 10
            return i + 15
        '\n        Send pickleData from the net on the local netMessenger.\n        The internal data in pickleData should have a tuple of\n        (messageString, sendArgsList).\n        '
        assert self.notify.debugCall()
        messageType = self.air.getMsgType()
        if messageType:
            message = MESSAGE_TYPES[messageType - 1]
            sentArgs = loads(pickleData)
        else:
            (message, sentArgs) = loads(pickleData)
        Messenger.send(self, message, sentArgs=sentArgs)