"""DNS Versioned Zones."""
import collections
import threading
from typing import Callable, Deque, Optional, Set, Union
import dns.exception
import dns.immutable
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.zone

class UseTransaction(dns.exception.DNSException):
    """To alter a versioned zone, use a transaction."""
Node = dns.zone.VersionedNode
ImmutableNode = dns.zone.ImmutableVersionedNode
Version = dns.zone.Version
WritableVersion = dns.zone.WritableVersion
ImmutableVersion = dns.zone.ImmutableVersion
Transaction = dns.zone.Transaction

class Zone(dns.zone.Zone):
    __slots__ = ['_versions', '_versions_lock', '_write_txn', '_write_waiters', '_write_event', '_pruning_policy', '_readers']
    node_factory = Node

    def __init__(self, origin: Optional[Union[dns.name.Name, str]], rdclass: dns.rdataclass.RdataClass=dns.rdataclass.IN, relativize: bool=True, pruning_policy: Optional[Callable[['Zone', Version], Optional[bool]]]=None):
        if False:
            print('Hello World!')
        "Initialize a versioned zone object.\n\n        *origin* is the origin of the zone.  It may be a ``dns.name.Name``,\n        a ``str``, or ``None``.  If ``None``, then the zone's origin will\n        be set by the first ``$ORIGIN`` line in a zone file.\n\n        *rdclass*, an ``int``, the zone's rdata class; the default is class IN.\n\n        *relativize*, a ``bool``, determine's whether domain names are\n        relativized to the zone's origin.  The default is ``True``.\n\n        *pruning policy*, a function taking a ``Zone`` and a ``Version`` and returning\n        a ``bool``, or ``None``.  Should the version be pruned?  If ``None``,\n        the default policy, which retains one version is used.\n        "
        super().__init__(origin, rdclass, relativize)
        self._versions: Deque[Version] = collections.deque()
        self._version_lock = threading.Lock()
        if pruning_policy is None:
            self._pruning_policy = self._default_pruning_policy
        else:
            self._pruning_policy = pruning_policy
        self._write_txn: Optional[Transaction] = None
        self._write_event: Optional[threading.Event] = None
        self._write_waiters: Deque[threading.Event] = collections.deque()
        self._readers: Set[Transaction] = set()
        self._commit_version_unlocked(None, WritableVersion(self, replacement=True), origin)

    def reader(self, id: Optional[int]=None, serial: Optional[int]=None) -> Transaction:
        if False:
            for i in range(10):
                print('nop')
        if id is not None and serial is not None:
            raise ValueError('cannot specify both id and serial')
        with self._version_lock:
            if id is not None:
                version = None
                for v in reversed(self._versions):
                    if v.id == id:
                        version = v
                        break
                if version is None:
                    raise KeyError('version not found')
            elif serial is not None:
                if self.relativize:
                    oname = dns.name.empty
                else:
                    assert self.origin is not None
                    oname = self.origin
                version = None
                for v in reversed(self._versions):
                    n = v.nodes.get(oname)
                    if n:
                        rds = n.get_rdataset(self.rdclass, dns.rdatatype.SOA)
                        if rds and rds[0].serial == serial:
                            version = v
                            break
                if version is None:
                    raise KeyError('serial not found')
            else:
                version = self._versions[-1]
            txn = Transaction(self, False, version)
            self._readers.add(txn)
            return txn

    def writer(self, replacement: bool=False) -> Transaction:
        if False:
            while True:
                i = 10
        event = None
        while True:
            with self._version_lock:
                if self._write_txn is None and event == self._write_event:
                    self._write_txn = Transaction(self, replacement, make_immutable=True)
                    self._write_event = None
                    break
                event = threading.Event()
                self._write_waiters.append(event)
            event.wait()
        self._write_txn._setup_version()
        return self._write_txn

    def _maybe_wakeup_one_waiter_unlocked(self):
        if False:
            print('Hello World!')
        if len(self._write_waiters) > 0:
            self._write_event = self._write_waiters.popleft()
            self._write_event.set()

    def _default_pruning_policy(self, zone, version):
        if False:
            return 10
        return True

    def _prune_versions_unlocked(self):
        if False:
            print('Hello World!')
        assert len(self._versions) > 0
        if len(self._readers) > 0:
            least_kept = min((txn.version.id for txn in self._readers))
        else:
            least_kept = self._versions[-1].id
        while self._versions[0].id < least_kept and self._pruning_policy(self, self._versions[0]):
            self._versions.popleft()

    def set_max_versions(self, max_versions: Optional[int]) -> None:
        if False:
            while True:
                i = 10
        'Set a pruning policy that retains up to the specified number\n        of versions\n        '
        if max_versions is not None and max_versions < 1:
            raise ValueError('max versions must be at least 1')
        if max_versions is None:

            def policy(zone, _):
                if False:
                    print('Hello World!')
                return False
        else:

            def policy(zone, _):
                if False:
                    for i in range(10):
                        print('nop')
                return len(zone._versions) > max_versions
        self.set_pruning_policy(policy)

    def set_pruning_policy(self, policy: Optional[Callable[['Zone', Version], Optional[bool]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the pruning policy for the zone.\n\n        The *policy* function takes a `Version` and returns `True` if\n        the version should be pruned, and `False` otherwise.  `None`\n        may also be specified for policy, in which case the default policy\n        is used.\n\n        Pruning checking proceeds from the least version and the first\n        time the function returns `False`, the checking stops.  I.e. the\n        retained versions are always a consecutive sequence.\n        '
        if policy is None:
            policy = self._default_pruning_policy
        with self._version_lock:
            self._pruning_policy = policy
            self._prune_versions_unlocked()

    def _end_read(self, txn):
        if False:
            return 10
        with self._version_lock:
            self._readers.remove(txn)
            self._prune_versions_unlocked()

    def _end_write_unlocked(self, txn):
        if False:
            return 10
        assert self._write_txn == txn
        self._write_txn = None
        self._maybe_wakeup_one_waiter_unlocked()

    def _end_write(self, txn):
        if False:
            i = 10
            return i + 15
        with self._version_lock:
            self._end_write_unlocked(txn)

    def _commit_version_unlocked(self, txn, version, origin):
        if False:
            for i in range(10):
                print('nop')
        self._versions.append(version)
        self._prune_versions_unlocked()
        self.nodes = version.nodes
        if self.origin is None:
            self.origin = origin
        if txn is not None:
            self._end_write_unlocked(txn)

    def _commit_version(self, txn, version, origin):
        if False:
            i = 10
            return i + 15
        with self._version_lock:
            self._commit_version_unlocked(txn, version, origin)

    def _get_next_version_id(self):
        if False:
            return 10
        if len(self._versions) > 0:
            id = self._versions[-1].id + 1
        else:
            id = 1
        return id

    def find_node(self, name: Union[dns.name.Name, str], create: bool=False) -> dns.node.Node:
        if False:
            return 10
        if create:
            raise UseTransaction
        return super().find_node(name)

    def delete_node(self, name: Union[dns.name.Name, str]) -> None:
        if False:
            print('Hello World!')
        raise UseTransaction

    def find_rdataset(self, name: Union[dns.name.Name, str], rdtype: Union[dns.rdatatype.RdataType, str], covers: Union[dns.rdatatype.RdataType, str]=dns.rdatatype.NONE, create: bool=False) -> dns.rdataset.Rdataset:
        if False:
            while True:
                i = 10
        if create:
            raise UseTransaction
        rdataset = super().find_rdataset(name, rdtype, covers)
        return dns.rdataset.ImmutableRdataset(rdataset)

    def get_rdataset(self, name: Union[dns.name.Name, str], rdtype: Union[dns.rdatatype.RdataType, str], covers: Union[dns.rdatatype.RdataType, str]=dns.rdatatype.NONE, create: bool=False) -> Optional[dns.rdataset.Rdataset]:
        if False:
            for i in range(10):
                print('nop')
        if create:
            raise UseTransaction
        rdataset = super().get_rdataset(name, rdtype, covers)
        if rdataset is not None:
            return dns.rdataset.ImmutableRdataset(rdataset)
        else:
            return None

    def delete_rdataset(self, name: Union[dns.name.Name, str], rdtype: Union[dns.rdatatype.RdataType, str], covers: Union[dns.rdatatype.RdataType, str]=dns.rdatatype.NONE) -> None:
        if False:
            return 10
        raise UseTransaction

    def replace_rdataset(self, name: Union[dns.name.Name, str], replacement: dns.rdataset.Rdataset) -> None:
        if False:
            i = 10
            return i + 15
        raise UseTransaction