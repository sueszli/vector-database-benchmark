from aim.ext.cleanup import AutoClean
from aim.ext.transport.client import Client

class RemoteResourceAutoClean(AutoClean):

    def __init__(self, instance):
        if False:
            i = 10
            return i + 15
        super().__init__(instance)
        self.hash = -1
        self.handler = None
        self.rpc_client: Client = None

    def _close(self):
        if False:
            while True:
                i = 10
        if self.handler is not None:
            assert self.rpc_client is not None
            self.rpc_client.release_resource(self.hash, self.handler)