import functools
from . import exceptions

def guarded(meth):
    if False:
        for i in range(10):
            print('nop')
    'A decorator to add a sanity check to ConnectionResource methods.'

    @functools.wraps(meth)
    def _check(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._check_conn_validity(meth.__name__)
        return meth(self, *args, **kwargs)
    return _check

class ConnectionResource:
    __slots__ = ('_connection', '_con_release_ctr')

    def __init__(self, connection):
        if False:
            return 10
        self._connection = connection
        self._con_release_ctr = connection._pool_release_ctr

    def _check_conn_validity(self, meth_name):
        if False:
            print('Hello World!')
        con_release_ctr = self._connection._pool_release_ctr
        if con_release_ctr != self._con_release_ctr:
            raise exceptions.InterfaceError('cannot call {}.{}(): the underlying connection has been released back to the pool'.format(self.__class__.__name__, meth_name))
        if self._connection.is_closed():
            raise exceptions.InterfaceError('cannot call {}.{}(): the underlying connection is closed'.format(self.__class__.__name__, meth_name))