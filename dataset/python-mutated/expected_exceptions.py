from nameko.rpc import rpc
from .auth import Auth, Unauthorized

class Service:
    name = 'service'
    auth = Auth()

    @rpc(expected_exceptions=Unauthorized)
    def update(self, data):
        if False:
            i = 10
            return i + 15
        if not self.auth.has_role('admin'):
            raise Unauthorized()
        raise TypeError('Whoops, genuine error.')