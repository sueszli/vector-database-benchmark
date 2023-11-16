from direct.distributed.DistributedObject import DistributedObject

class AIDGameObject(DistributedObject):
    """ This class is a DirectObject which will be created and managed by the
    AI Repository. """

    def __init__(self, cr):
        if False:
            for i in range(10):
                print('nop')
        DistributedObject.__init__(self, cr)

    def announceGenerate(self):
        if False:
            i = 10
            return i + 15
        " The AI has created this object, so we send it's distributed object ID\n        over to the client.  That way the client can actually grab the object\n        and use it to communicate with the AI.  Alternatively store it in the\n        Client Repository in self.cr "
        base.messenger.send(self.cr.uniqueName('AIDGameObjectGenerated'), [self.doId])
        DistributedObject.announceGenerate(self)

    def d_requestDataFromAI(self):
        if False:
            print('Hello World!')
        ' Request some data from the AI and passing it some data from us. '
        data = ('Some Data', 1, -1.25)
        print('Sending game data:', data)
        self.sendUpdate('messageRoundtripToAI', [data])

    def messageRoundtripToClient(self, data):
        if False:
            for i in range(10):
                print('nop')
        ' Here we expect the answer from the AI from a previous\n        messageRoundtripToAI call '
        print('Got Data:', data)
        print('Roundtrip message complete')