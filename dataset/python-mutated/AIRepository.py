from direct.distributed.ClientRepository import ClientRepository
from panda3d.core import URLSpec, ConfigVariableInt, ConfigVariableString

class AIRepository(ClientRepository):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        ' The AI Repository usually lives on a server and is responsible for\n        server side logic that will handle game objects '
        dcFileNames = ['../direct.dc', 'sample.dc']
        ClientRepository.__init__(self, dcFileNames=dcFileNames, dcSuffix='AI', threadedNet=True)
        tcpPort = ConfigVariableInt('server-port', 4400).getValue()
        hostname = ConfigVariableString('server-host', '127.0.0.1').getValue()
        url = URLSpec('http://{}:{}'.format(hostname, tcpPort))
        self.connect([url], successCallback=self.connectSuccess, failureCallback=self.connectFailure)

    def connectFailure(self, statusCode, statusString):
        if False:
            while True:
                i = 10
        ' something went wrong '
        print("Couldn't connect. Make sure to run server.py first!")
        raise (StandardError, statusString)

    def connectSuccess(self):
        if False:
            while True:
                i = 10
        " Successfully connected.  But we still can't really do\n        anything until we've got the doID range. "
        self.accept('createReady', self.gotCreateReady)

    def lostConnection(self):
        if False:
            while True:
                i = 10
        ' This should be overridden by a derived class to handle an\n         unexpectedly lost connection to the gameserver. '
        exit()

    def gotCreateReady(self):
        if False:
            while True:
                i = 10
        " Now we're ready to go! "
        if not self.haveCreateAuthority():
            return
        self.ignore('createReady')
        self.timeManager = self.createDistributedObject(className='TimeManagerAI', zoneId=1)
        print('AI Repository Ready')

    def deallocateChannel(self, doID):
        if False:
            while True:
                i = 10
        ' This method will be called whenever a client disconnects from the\n        server.  The given doID is the ID of the client who left us. '
        print('Client left us: ', doID)