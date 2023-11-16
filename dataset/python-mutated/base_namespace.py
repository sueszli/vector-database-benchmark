class BaseNamespace(object):

    def __init__(self, namespace=None):
        if False:
            return 10
        self.namespace = namespace or '/'

    def is_asyncio_based(self):
        if False:
            while True:
                i = 10
        return False

class BaseServerNamespace(BaseNamespace):

    def __init__(self, namespace=None):
        if False:
            print('Hello World!')
        super().__init__(namespace=namespace)
        self.server = None

    def _set_server(self, server):
        if False:
            while True:
                i = 10
        self.server = server

    def rooms(self, sid, namespace=None):
        if False:
            print('Hello World!')
        'Return the rooms a client is in.\n\n        The only difference with the :func:`socketio.Server.rooms` method is\n        that when the ``namespace`` argument is not given the namespace\n        associated with the class is used.\n        '
        return self.server.rooms(sid, namespace=namespace or self.namespace)

class BaseClientNamespace(BaseNamespace):

    def __init__(self, namespace=None):
        if False:
            return 10
        super().__init__(namespace=namespace)
        self.client = None

    def _set_client(self, client):
        if False:
            return 10
        self.client = client