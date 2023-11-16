from twisted.cred import credentials
from twisted.internet import reactor
from twisted.spread import pb

class UsersClient:
    """
    Client set up in buildbot.scripts.runner to send `buildbot user` args
    over a PB connection to perspective_commandline that will execute the
    args on the database.
    """

    def __init__(self, master, username, password, port):
        if False:
            i = 10
            return i + 15
        self.host = master
        self.username = username
        self.password = password
        self.port = int(port)

    def send(self, op, bb_username, bb_password, ids, info):
        if False:
            print('Hello World!')
        f = pb.PBClientFactory()
        d = f.login(credentials.UsernamePassword(self.username, self.password))
        reactor.connectTCP(self.host, self.port, f)

        @d.addCallback
        def call_commandline(remote):
            if False:
                for i in range(10):
                    print('nop')
            d = remote.callRemote('commandline', op, bb_username, bb_password, ids, info)

            @d.addCallback
            def returnAndLose(res):
                if False:
                    print('Hello World!')
                remote.broker.transport.loseConnection()
                return res
            return d
        return d