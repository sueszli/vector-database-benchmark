"""Implements a shared object that spans processes.

This object will be instanciated once per VM and methods will be invoked
on it via rpc.
"""
import logging
import multiprocessing.managers
import os
import tempfile
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Optional
from typing import TypeVar
import fasteners
autoproxy = multiprocessing.managers.AutoProxy

def patched_autoproxy(token, serializer, manager=None, authkey=None, exposed=None, incref=True, manager_owned=True):
    if False:
        return 10
    return autoproxy(token, serializer, manager, authkey, exposed, incref)
multiprocessing.managers.AutoProxy = patched_autoproxy
T = TypeVar('T')
AUTH_KEY = b'mps'

class _SingletonProxy:
    """Proxies the shared object so we can release it with better errors and no
  risk of dangling references in the multiprocessing manager infrastructure.
  """

    def __init__(self, entry):
        if False:
            while True:
                i = 10
        self._SingletonProxy_entry = entry
        self._SingletonProxy_valid = True

    def singletonProxy_call__(self, *args, **kwargs):
        if False:
            return 10
        if not self._SingletonProxy_valid:
            raise RuntimeError('Entry was released.')
        return self._SingletonProxy_entry.obj.__call__(*args, **kwargs)

    def singletonProxy_release(self):
        if False:
            return 10
        assert self._SingletonProxy_valid
        self._SingletonProxy_valid = False

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if not self._SingletonProxy_valid:
            raise RuntimeError('Entry was released.')
        try:
            return getattr(self._SingletonProxy_entry.obj, name)
        except AttributeError as e:
            logging.info('Attribute %s is unavailable as a public function because its __getattr__ function raised the following exception %s', name, e)
            return None

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        dir = self._SingletonProxy_entry.obj.__dir__()
        dir.append('singletonProxy_call__')
        dir.append('singletonProxy_release')
        return dir

class _SingletonEntry:
    """Represents a single, refcounted entry in this process."""

    def __init__(self, constructor, initialize_eagerly=True):
        if False:
            for i in range(10):
                print('nop')
        self.constructor = constructor
        self.refcount = 0
        self.lock = threading.Lock()
        if initialize_eagerly:
            self.obj = constructor()
            self.initialied = True
        else:
            self.initialied = False

    def acquire(self):
        if False:
            i = 10
            return i + 15
        with self.lock:
            if not self.initialied:
                self.obj = self.constructor()
                self.initialied = True
            self.refcount += 1
            return _SingletonProxy(self)

    def release(self, proxy):
        if False:
            i = 10
            return i + 15
        proxy.singletonProxy_release()
        with self.lock:
            self.refcount -= 1
            if self.refcount == 0:
                del self.obj
                self.initialied = False

class _SingletonManager:
    entries: Dict[Any, Any] = {}

    def register_singleton(self, constructor, tag, initialize_eagerly=True):
        if False:
            print('Hello World!')
        assert tag not in self.entries, tag
        self.entries[tag] = _SingletonEntry(constructor, initialize_eagerly)

    def has_singleton(self, tag):
        if False:
            print('Hello World!')
        return tag in self.entries

    def acquire_singleton(self, tag):
        if False:
            i = 10
            return i + 15
        return self.entries[tag].acquire()

    def release_singleton(self, tag, obj):
        if False:
            i = 10
            return i + 15
        return self.entries[tag].release(obj)
_process_level_singleton_manager = _SingletonManager()
_process_local_lock = threading.Lock()

class _SingletonRegistrar(multiprocessing.managers.BaseManager):
    pass
_SingletonRegistrar.register('acquire_singleton', callable=_process_level_singleton_manager.acquire_singleton)
_SingletonRegistrar.register('release_singleton', callable=_process_level_singleton_manager.release_singleton)

class _AutoProxyWrapper:

    def __init__(self, proxyObject: multiprocessing.managers.BaseProxy):
        if False:
            while True:
                i = 10
        self._proxyObject = proxyObject

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._proxyObject.singletonProxy_call__(*args, **kwargs)

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return getattr(self._proxyObject, name)

    def get_auto_proxy_object(self):
        if False:
            return 10
        return self._proxyObject

class MultiProcessShared(Generic[T]):
    """MultiProcessShared is used to share a single object across processes.

  For example, one could have the class::

    class MyExpensiveObject(object):
      def __init__(self, args):
        [expensive initialization and memory allocation]

      def method(self, arg):
        ...

  One could share a single instance of this class by wrapping it as::

    shared_ptr = MultiProcessShared(lambda: MyExpensiveObject(...))
    my_expensive_object = shared_ptr.acquire()

  which could then be invoked as::

    my_expensive_object.method(arg)

  This can then be released with::

    shared_ptr.release(my_expensive_object)

  but care should be taken to avoid releasing the object too soon or
  expensive re-initialization may be required, defeating the point of
  using a shared object.


  Args:
    constructor: function that initialises / constructs the object if not
      present in the cache. This function should take no arguments. It should
      return an initialised object, or raise an exception if the object could
      not be initialised / constructed.
    tag: an indentifier to store with the cached object. If multiple
      MultiProcessShared instances are created with the same tag, they will all
      share the same proxied object.
    path: a temporary path in which to create the inter-process lock
    always_proxy: whether to direct all calls through the proxy, rather than
      call the object directly for the process that created it
  """

    def __init__(self, constructor: Callable[[], T], tag: Any, *, path: str=tempfile.gettempdir(), always_proxy: Optional[bool]=None):
        if False:
            while True:
                i = 10
        self._constructor = constructor
        self._tag = tag
        self._path = path
        self._always_proxy = False if always_proxy is None else always_proxy
        self._proxy = None
        self._manager = None
        self._rpc_address = None
        self._cross_process_lock = fasteners.InterProcessLock(os.path.join(self._path, self._tag) + '.lock')

    def _get_manager(self):
        if False:
            return 10
        if self._manager is None:
            address_file = os.path.join(self._path, self._tag) + '.address'
            while self._manager is None:
                with _process_local_lock:
                    with self._cross_process_lock:
                        if not os.path.exists(address_file):
                            self._create_server(address_file)
                        if _process_level_singleton_manager.has_singleton(self._tag) and (not self._always_proxy):
                            self._manager = _process_level_singleton_manager
                        else:
                            with open(address_file) as fin:
                                address = fin.read()
                            logging.info('Connecting to remote proxy at %s', address)
                            (host, port) = address.split(':')
                            manager = _SingletonRegistrar(address=(host, int(port)), authkey=AUTH_KEY)
                            multiprocessing.current_process().authkey = AUTH_KEY
                            try:
                                manager.connect()
                                self._manager = manager
                            except ConnectionError:
                                os.unlink(address_file)
        return self._manager

    def acquire(self):
        if False:
            for i in range(10):
                print('nop')
        singleton = self._get_manager().acquire_singleton(self._tag)
        return _AutoProxyWrapper(singleton)

    def release(self, obj):
        if False:
            i = 10
            return i + 15
        self._manager.release_singleton(self._tag, obj.get_auto_proxy_object())

    def _create_server(self, address_file):
        if False:
            for i in range(10):
                print('nop')
        self._serving_manager = _SingletonRegistrar(address=('localhost', 0), authkey=AUTH_KEY)
        multiprocessing.current_process().authkey = AUTH_KEY
        _process_level_singleton_manager.register_singleton(self._constructor, self._tag, initialize_eagerly=True)
        self._server = self._serving_manager.get_server()
        logging.info('Starting proxy server at %s for shared %s', self._server.address, self._tag)
        with open(address_file + '.tmp', 'w') as fout:
            fout.write('%s:%d' % self._server.address)
        os.rename(address_file + '.tmp', address_file)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        logging.info('Done starting server')