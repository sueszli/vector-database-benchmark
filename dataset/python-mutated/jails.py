__author__ = 'Cyril Jaquier, Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier, 2013- Yaroslav Halchenko'
__license__ = 'GPL'
from threading import Lock
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from ..exceptions import DuplicateJailException, UnknownJailException
from .jail import Jail

class Jails(Mapping):
    """Handles the jails.

	This class handles the jails. Creation, deletion or access to a jail
	must be done through this class. This class is thread-safe which is
	not the case of the jail itself, including filter and actions. This
	class is based on Mapping type, and the `add` method must be used to
	add additional jails.
	"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__lock = Lock()
        self._jails = dict()

    def add(self, name, backend, db=None):
        if False:
            while True:
                i = 10
        "Adds a jail.\n\n\t\tAdds a new jail if not already present which should use the\n\t\tgiven backend.\n\n\t\tParameters\n\t\t----------\n\t\tname : str\n\t\t\tThe name of the jail.\n\t\tbackend : str\n\t\t\tThe backend to use.\n\t\tdb : Fail2BanDb\n\t\t\tFail2Ban's persistent database instance.\n\n\t\tRaises\n\t\t------\n\t\tDuplicateJailException\n\t\t\tIf jail name is already present.\n\t\t"
        with self.__lock:
            if name in self._jails:
                raise DuplicateJailException(name)
            else:
                self._jails[name] = Jail(name, backend, db)

    def exists(self, name):
        if False:
            while True:
                i = 10
        return name in self._jails

    def __getitem__(self, name):
        if False:
            return 10
        try:
            self.__lock.acquire()
            return self._jails[name]
        except KeyError:
            raise UnknownJailException(name)
        finally:
            self.__lock.release()

    def __delitem__(self, name):
        if False:
            return 10
        try:
            self.__lock.acquire()
            del self._jails[name]
        except KeyError:
            raise UnknownJailException(name)
        finally:
            self.__lock.release()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        try:
            self.__lock.acquire()
            return len(self._jails)
        finally:
            self.__lock.release()

    def __iter__(self):
        if False:
            while True:
                i = 10
        try:
            self.__lock.acquire()
            return iter(self._jails)
        finally:
            self.__lock.release()