from direct.distributed.ClientRepository import ClientRepository
from panda3d.core import URLSpec, ConfigVariableInt, ConfigVariableString
from DistributedSmoothActor import DistributedSmoothActor

class GameClientRepository(ClientRepository):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        dcFileNames = ['../direct.dc', 'sample.dc']
        self.distributedObject = None
        self.aiDGameObect = None
        ClientRepository.__init__(self, dcFileNames=dcFileNames, threadedNet=True)
        tcpPort = ConfigVariableInt('server-port', 4400).getValue()
        hostname = ConfigVariableString('server-host', '127.0.0.1').getValue()
        self.url = URLSpec('http://{}:{}'.format(hostname, tcpPort))
        self.connect([self.url], successCallback=self.connectSuccess, failureCallback=self.connectFailure)

    def lostConnection(self):
        if False:
            while True:
                i = 10
        ' This should be overridden by a derived class to handle an\n        unexpectedly lost connection to the gameserver. '
        exit()

    def connectFailure(self, statusCode, statusString):
        if False:
            i = 10
            return i + 15
        ' Something went wrong '
        exit()

    def connectSuccess(self):
        if False:
            return 10
        " Successfully connected.  But we still can't really do\n        anything until we've got the doID range. "
        self.setInterestZones([1])
        self.acceptOnce(self.uniqueName('gotTimeSync'), self.syncReady)

    def syncReady(self):
        if False:
            while True:
                i = 10
        " Now we've got the TimeManager manifested, and we're in\n        sync with the server time.  Now we can enter the world.  Check\n        to see if we've received our doIdBase yet. "
        if self.haveCreateAuthority():
            self.gotCreateReady()
        else:
            self.accept(self.uniqueName('createReady'), self.gotCreateReady)

    def gotCreateReady(self):
        if False:
            return 10
        ' Ready to enter the world.  Expand our interest to include\n        any other zones '
        if not self.haveCreateAuthority():
            return
        self.ignore(self.uniqueName('createReady'))
        self.join()
        print('Client Ready')

    def join(self):
        if False:
            while True:
                i = 10
        ' Join a game/room/whatever '
        self.setInterestZones([1, 2])
        base.messenger.send('client-joined')
        print('Joined')