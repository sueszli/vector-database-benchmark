import struct
from twisted.internet import reactor
from twisted.internet import task
from twisted.internet.protocol import DatagramProtocol
PORT = 51444
message = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
active = False

def timeout():
    if False:
        i = 10
        return i + 15
    global active, message
    active = False
    if not active:
        message = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class twistedUDP(DatagramProtocol):

    def datagramReceived(self, data, addr):
        if False:
            for i in range(10):
                print('nop')
        global message, active
        active = True
        numOfValues = len(data) // 8
        mess = struct.unpack('>' + 'd' * numOfValues, data)
        message = [round(element, 3) for element in mess]

def startTwisted():
    if False:
        i = 10
        return i + 15
    l = task.LoopingCall(timeout)
    l.start(0.5)
    reactor.listenUDP(PORT, twistedUDP())
    reactor.run(installSignalHandlers=False)