from direct.distributed.DistributedObject import DistributedObject

class DGameObject(DistributedObject):

    def __init__(self, cr):
        if False:
            while True:
                i = 10
        DistributedObject.__init__(self, cr)

    def sendGameData(self, data):
        if False:
            return 10
        ' Method that can be called from the clients with an sendUpdate call '
        print(data)

    def d_sendGameData(self):
        if False:
            for i in range(10):
                print('nop')
        ' A method to send an update message to the server.  The d_ stands\n        for distributed '
        self.sendUpdate('sendGameData', [('ValueA', 123, 1.25)])