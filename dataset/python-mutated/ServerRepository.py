from direct.distributed.ServerRepository import ServerRepository
from panda3d.core import ConfigVariableInt

class GameServerRepository(ServerRepository):
    """The server repository class"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'initialise the server class'
        tcpPort = ConfigVariableInt('server-port', 4400).getValue()
        dcFileNames = ['../direct.dc', 'sample.dc']
        ServerRepository.__init__(self, tcpPort, dcFileNames=dcFileNames, threadedNet=True)