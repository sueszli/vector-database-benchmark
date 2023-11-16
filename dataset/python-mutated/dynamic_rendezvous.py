import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import NodeState, construct_and_record_rdzv_event
from .api import RendezvousClosedError, RendezvousError, RendezvousHandler, RendezvousParameters, RendezvousStateError, RendezvousTimeoutError
from .utils import _delay, _PeriodicTimer
__all__ = ['RendezvousBackend', 'RendezvousTimeout', 'RendezvousSettings', 'DynamicRendezvousHandler', 'create_handler']
log = logging.getLogger(__name__)

def get_method_name(depth=2):
    if False:
        return 10
    if len(inspect.stack()) > depth:
        return inspect.stack()[depth].function
    return 'no_method_name'
Token = Any
'Represent an opaque fencing token used by the rendezvous backend.'

class RendezvousBackend(ABC):
    """Represent a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get the name of the backend.'

    @abstractmethod
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        if False:
            for i in range(10):
                print('nop')
        'Get the rendezvous state.\n\n        Returns:\n            A tuple of the encoded rendezvous state and its fencing token or\n            ``None`` if no state is found in the backend.\n\n        Raises:\n            RendezvousConnectionError:\n                The connection to the backend has failed.\n            RendezvousStateError:\n                The rendezvous state is corrupt.\n        '

    @abstractmethod
    def set_state(self, state: bytes, token: Optional[Token]=None) -> Optional[Tuple[bytes, Token, bool]]:
        if False:
            return 10
        "Set the rendezvous state.\n\n        The new rendezvous state is set conditionally:\n\n          - If the specified ``token`` matches the fencing token stored in the\n            backend, the state will be updated. The new state will be returned\n            to the caller along with its fencing token.\n          - If the specified ``token`` does not match the fencing token stored\n            in the backend, the state won't be updated; instead the existing\n            state along with its fencing token will be returned to the caller.\n          - If the specified ``token`` is ``None``, the new state will be set\n            only if there is no existing state in the backend. Either the new\n            state or the existing state along with its fencing token will be\n            returned to the caller.\n\n        Args:\n            state:\n                The encoded rendezvous state.\n            token:\n                An optional fencing token that was retrieved by a previous call\n                to :py:meth:`get_state` or ``set_state()``.\n\n        Returns:\n            A tuple of the serialized rendezvous state, its fencing token, and\n            a boolean value indicating whether our set attempt succeeded.\n\n        Raises:\n            RendezvousConnectionError:\n                The connection to the backend has failed.\n            RendezvousStateError:\n                The rendezvous state is corrupt.\n        "

class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        keep_alive:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """
    _ZERO = timedelta(0)
    _DEFAULT_TIMEOUTS = {'join': timedelta(seconds=600), 'last_call': timedelta(seconds=30), 'close': timedelta(seconds=30), 'heartbeat': timedelta(seconds=5)}
    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta

    def __init__(self, join: Optional[timedelta]=None, last_call: Optional[timedelta]=None, close: Optional[timedelta]=None, heartbeat: Optional[timedelta]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._set_timeouts(join=join, last_call=last_call, close=close, heartbeat=heartbeat)

    @property
    def join(self) -> timedelta:
        if False:
            return 10
        'Get the join timeout.'
        return self._join

    @property
    def last_call(self) -> timedelta:
        if False:
            return 10
        'Get the last call timeout.'
        return self._last_call

    @property
    def close(self) -> timedelta:
        if False:
            print('Hello World!')
        'Get the close timeout.'
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        if False:
            for i in range(10):
                print('nop')
        'Get the keep-alive heartbeat timeout.'
        return self._heartbeat

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        if False:
            print('Hello World!')
        for (name, timeout) in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f'The {name} timeout ({timeout}) must be positive.')
            setattr(self, '_' + name, timeout)

@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Hold the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """
    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int

@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """
    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.addr}_{self.pid}_{self.local_id}'

class _NodeDescGenerator:
    """Generate node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """
    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._lock = threading.Lock()
        self._local_id = 0

    def generate(self, local_addr: Optional[str]=None) -> _NodeDesc:
        if False:
            while True:
                i = 10
        with self._lock:
            local_id = self._local_id
            self._local_id += 1
        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)

class _RendezvousState:
    """Hold the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """
    round: int
    complete: bool
    deadline: Optional[datetime]
    closed: bool
    participants: Dict[_NodeDesc, int]
    wait_list: Set[_NodeDesc]
    last_heartbeats: Dict[_NodeDesc, datetime]

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.round = 0
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.last_heartbeats = {}

def _remove_participant_epilogue(state: _RendezvousState, settings: RendezvousSettings) -> None:
    if False:
        for i in range(10):
            print('nop')
    if state.complete:
        if not state.participants:
            state.complete = False
            state.round += 1
    elif len(state.participants) < settings.min_nodes:
        state.deadline = None

class _RendezvousStateHolder(ABC):
    """Hold the shared rendezvous state synced with other nodes."""

    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        if False:
            i = 10
            return i + 15
        'Get the local state.'

    @abstractmethod
    def sync(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        'Read or writes the latest state.\n\n        Returns:\n            A boolean value indicating whether the local state, in case marked\n            as dirty, was successfully synced with other nodes.\n        '

    @abstractmethod
    def mark_dirty(self) -> None:
        if False:
            while True:
                i = 10
        'Mark the local state as dirty.'

class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    """Hold the rendezvous state synced with other nodes via a backend.

    Args:
        backend:
            The rendezvous backend to use.
        settings:
            The rendezvous settings.
        cache_duration:
            The amount of time, in seconds, to cache the last rendezvous state
            before requesting it from the backend again.
    """
    _backend: RendezvousBackend
    _state: _RendezvousState
    _settings: RendezvousSettings
    _cache_duration: int
    _token: Token
    _dirty: bool
    _last_sync_time: float
    _dead_nodes: List[_NodeDesc]

    def __init__(self, backend: RendezvousBackend, settings: RendezvousSettings, cache_duration: int=1) -> None:
        if False:
            i = 10
            return i + 15
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = cache_duration
        self._token = None
        self._dirty = False
        self._last_sync_time = -1
        self._dead_nodes = []

    def _record(self, message: str, node_state: NodeState=NodeState.RUNNING):
        if False:
            for i in range(10):
                print('nop')
        construct_and_record_rdzv_event(name=f'{self.__class__.__name__}.{get_method_name()}', run_id=self._settings.run_id, message=message, node_state=node_state)

    @property
    def state(self) -> _RendezvousState:
        if False:
            while True:
                i = 10
        'See base class.'
        return self._state

    def sync(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        state_bits: Optional[bytes] = None
        token = None
        has_set: Optional[bool]
        if self._dirty:
            has_set = False
            state_bits = pickle.dumps(self._state)
            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                (state_bits, token, has_set) = set_response
        else:
            has_set = None
            if self._cache_duration > 0:
                if self._last_sync_time >= max(time.monotonic() - self._cache_duration, 0):
                    return None
            get_response = self._backend.get_state()
            if get_response is not None:
                (state_bits, token) = get_response
        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                raise RendezvousStateError('The rendezvous state is corrupt. See inner exception for details.') from exc
        else:
            self._state = _RendezvousState()
        if has_set and self._dead_nodes and log.isEnabledFor(logging.DEBUG):
            node_list = ', '.join((f"'{dead_node}'" for dead_node in self._dead_nodes))
            msg = f"As part of the sync operation the node(s) {node_list} have been removed from the rendezvous '{self._settings.run_id}' since they had no heartbeat."
            self._record(message=msg)
            log.debug(msg)
        self._token = token
        self._dirty = False
        self._last_sync_time = time.monotonic()
        self._sanitize()
        return has_set

    def _sanitize(self) -> None:
        if False:
            print('Hello World!')
        state = self._state
        expire_time = datetime.utcnow() - self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        self._dead_nodes = [node for (node, last_heartbeat) in state.last_heartbeats.items() if last_heartbeat < expire_time]
        participant_removed = False
        for dead_node in self._dead_nodes:
            del state.last_heartbeats[dead_node]
            try:
                del state.participants[dead_node]
                participant_removed = True
            except KeyError:
                pass
            try:
                state.wait_list.remove(dead_node)
            except KeyError:
                pass
        if participant_removed:
            _remove_participant_epilogue(state, self._settings)

    def mark_dirty(self) -> None:
        if False:
            print('Hello World!')
        'See base class.\n\n        If the local rendezvous state is dirty, the next sync call will try to\n        write the changes back to the backend. However this attempt might fail\n        if another node, which had the same state, also made changes and wrote\n        them before us.\n        '
        self._dirty = True

class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""
    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    REMOVE_FROM_PARTICIPANTS = 4
    REMOVE_FROM_WAIT_LIST = 5
    MARK_RENDEZVOUS_COMPLETE = 6
    MARK_RENDEZVOUS_CLOSED = 7
    SYNC = 8
    ERROR_CLOSED = 9
    ERROR_TIMEOUT = 10
    FINISH = 11

class _RendezvousContext:
    """Holds the context of the rendezvous.

    Attributes:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state:
            The current state of the rendezvous.
        settings:
            The rendezvous settings.
    """
    node: _NodeDesc
    state: _RendezvousState
    settings: RendezvousSettings

    def __init__(self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.node = node
        self.state = state
        self.settings = settings

class _RendezvousOpExecutor(ABC):
    """Execute rendezvous operations."""

    @abstractmethod
    def run(self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float) -> None:
        if False:
            print('Hello World!')
        'Execute a rendezvous operation.\n\n        An operation is run inside a state machine and is expected to transition\n        the rendezvous from one state to another.\n\n        Args:\n            state_handler:\n                A callable that is expected to return the next state transition\n                action based on the current state of the rendezvous.\n            deadline:\n                The time, in seconds, at which the operation will be considered\n                timed-out.\n        '

class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Execute rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """
    _node: _NodeDesc
    _state: _RendezvousState
    _state_holder: _RendezvousStateHolder
    _settings: RendezvousSettings

    def __init__(self, node: _NodeDesc, state_holder: _RendezvousStateHolder, settings: RendezvousSettings) -> None:
        if False:
            print('Hello World!')
        self._node = node
        self._state_holder = state_holder
        self._settings = settings

    def _record(self, message: str, node_state: NodeState=NodeState.RUNNING) -> None:
        if False:
            i = 10
            return i + 15
        construct_and_record_rdzv_event(name=f'{self.__class__.__name__}.{get_method_name()}', run_id=self._settings.run_id, message=message, node_state=node_state, hostname=self._node.addr, pid=self._node.pid, local_id=self._node.local_id)

    def run(self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float) -> None:
        if False:
            return 10
        'See base class.'
        action = None
        while action != _Action.FINISH:
            has_set = self._state_holder.sync()
            if has_set is not None:
                if has_set:
                    msg = f"The node '{self._node}' has successfully synced its local changes with other nodes in the rendezvous '{self._settings.run_id}'."
                else:
                    msg = f"The node '{self._node}' has a stale state and failed to sync its local changes with other nodes in the rendezvous '{self._settings.run_id}'."
                self._record(message=msg)
                log.debug(msg)
            self._state = self._state_holder.state
            ctx = _RendezvousContext(self._node, self._state, self._settings)
            action = state_handler(ctx, deadline)
            if action == _Action.FINISH:
                continue
            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError()
            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError()
            if action == _Action.SYNC:
                _delay(seconds=1)
            else:
                if action == _Action.KEEP_ALIVE:
                    self._keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:
                    self._add_to_wait_list()
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()
                self._state_holder.mark_dirty()

    def _keep_alive(self) -> None:
        if False:
            print('Hello World!')
        msg = f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        self._state.last_heartbeats[self._node] = datetime.utcnow()

    def _add_to_participants(self) -> None:
        if False:
            return 10
        msg = f"The node '{self._node}' added itself to the participants of round {self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        state = self._state
        try:
            state.wait_list.remove(self._node)
        except KeyError:
            pass
        state.participants[self._node] = 0
        self._keep_alive()
        if len(state.participants) == self._settings.min_nodes:
            state.deadline = datetime.utcnow() + self._settings.timeout.last_call
        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    def _add_to_wait_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        msg = f"The node '{self._node}' added itself to the wait list of round {self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        self._state.wait_list.add(self._node)
        self._keep_alive()

    def _remove_from_participants(self) -> None:
        if False:
            while True:
                i = 10
        msg = f"The node '{self._node}' removed itself from the participants of round {self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        state = self._state
        del state.participants[self._node]
        del state.last_heartbeats[self._node]
        _remove_participant_epilogue(state, self._settings)

    def _remove_from_wait_list(self) -> None:
        if False:
            return 10
        msg = f"The node '{self._node}' removed itself from the wait list of round {self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        self._state.wait_list.remove(self._node)
        del self._state.last_heartbeats[self._node]

    def _mark_rendezvous_complete(self) -> None:
        if False:
            i = 10
            return i + 15
        msg = f"The node '{self._node}' marked round {self._state.round} of the rendezvous '{self._settings.run_id}' as complete. Pending sync."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.debug(msg)
        state = self._state
        state.complete = True
        state.deadline = None
        for (rank, node) in enumerate(sorted(state.participants)):
            state.participants[node] = rank

    def _mark_rendezvous_closed(self) -> None:
        if False:
            print('Hello World!')
        msg = f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. Pending sync."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.debug(msg)
        self._state.closed = True

def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Determine whether a keep-alive heartbeat should be sent.'
    try:
        last_heartbeat = ctx.state.last_heartbeats[ctx.node]
    except KeyError:
        return False
    return last_heartbeat <= datetime.utcnow() - ctx.settings.keep_alive_interval

class _RendezvousExitOp:
    """Represent a rendezvous exit operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if False:
            while True:
                i = 10
        if ctx.node in ctx.state.participants:
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.REMOVE_FROM_PARTICIPANTS
        return _Action.FINISH

class _RendezvousJoinOp:
    """Represent a rendezvous join operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if False:
            return 10
        state = ctx.state
        if state.closed:
            return _Action.ERROR_CLOSED
        is_participant = ctx.node in state.participants
        if state.complete and is_participant:
            return _Action.FINISH
        now = time.monotonic()
        if now > deadline:
            rollback_period = 5
            if now <= deadline + rollback_period:
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                if ctx.node in state.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT
        if state.complete:
            if len(state.participants) < ctx.settings.max_nodes:
                if ctx.node not in state.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
        elif is_participant:
            if len(state.participants) >= ctx.settings.min_nodes:
                if cast(datetime, state.deadline) < datetime.utcnow():
                    return _Action.MARK_RENDEZVOUS_COMPLETE
        else:
            return _Action.ADD_TO_PARTICIPANTS
        if _should_keep_alive(ctx):
            return _Action.KEEP_ALIVE
        return _Action.SYNC

class _RendezvousCloseOp:
    """Represent a rendezvous close operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if False:
            i = 10
            return i + 15
        if ctx.state.closed:
            return _Action.FINISH
        if time.monotonic() > deadline:
            return _Action.ERROR_TIMEOUT
        return _Action.MARK_RENDEZVOUS_CLOSED

class _RendezvousKeepAliveOp:
    """Represent a rendezvous keep-alive update operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if False:
            return 10
        if _should_keep_alive(ctx):
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.KEEP_ALIVE
        return _Action.FINISH

class DynamicRendezvousHandler(RendezvousHandler):
    """Represent a handler that sets up a rendezvous among a set of nodes."""
    _node_desc_generator = _NodeDescGenerator()
    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: Optional[_PeriodicTimer]

    @classmethod
    def from_backend(cls, run_id: str, store: Store, backend: RendezvousBackend, min_nodes: int, max_nodes: int, local_addr: Optional[str]=None, timeout: Optional[RendezvousTimeout]=None):
        if False:
            i = 10
            return i + 15
        'Create a new :py:class:`DynamicRendezvousHandler`.\n\n        Args:\n            run_id:\n                The run id of the rendezvous.\n            store:\n                The C10d store to return as part of the rendezvous.\n            backend:\n                The backend to use to hold the rendezvous state.\n            min_nodes:\n                The minimum number of nodes to admit to the rendezvous.\n            max_nodes:\n                The maximum number of nodes to admit to the rendezvous.\n            local_addr:\n                The local node address.\n            timeout:\n                The timeout configuration of the rendezvous.\n        '
        node = cls._node_desc_generator.generate(local_addr)
        settings = RendezvousSettings(run_id, min_nodes, max_nodes, timeout or RendezvousTimeout(), keep_alive_interval=timedelta(seconds=5), keep_alive_max_attempt=3)
        state_holder = _BackendRendezvousStateHolder(backend, settings)
        return cls(node, settings, backend.name, store, state_holder)

    def __init__(self, node: _NodeDesc, settings: RendezvousSettings, backend_name: str, store: Store, state_holder: _RendezvousStateHolder) -> None:
        if False:
            i = 10
            return i + 15
        if not settings.run_id:
            raise ValueError('The run id must be a non-empty string.')
        if settings.min_nodes < 1:
            raise ValueError(f'The minimum number of nodes ({settings.min_nodes}) must be greater than zero.')
        if settings.max_nodes < settings.min_nodes:
            raise ValueError(f'The maximum number of nodes ({settings.max_nodes}) must be greater than or equal to the minimum number of nodes ({settings.min_nodes}).')
        self._this_node = node
        self._settings = settings
        self._backend_name = backend_name
        self._store = store
        self._state_holder = state_holder
        self._op_executor = _DistributedRendezvousOpExecutor(self._this_node, self._state_holder, self._settings)
        self._heartbeat_lock = threading.Lock()
        self._keep_alive_timer = None

    def _record(self, message: str, node_state: NodeState=NodeState.RUNNING, rank: Optional[int]=None) -> None:
        if False:
            return 10
        construct_and_record_rdzv_event(name=f'{self.__class__.__name__}.{get_method_name()}', run_id=self._settings.run_id, message=message, node_state=node_state, hostname=self._this_node.addr, pid=self._this_node.pid, local_id=self._this_node.local_id, rank=rank)

    @property
    def settings(self) -> RendezvousSettings:
        if False:
            for i in range(10):
                print('nop')
        'Get the settings of the rendezvous.'
        return self._settings

    def get_backend(self) -> str:
        if False:
            while True:
                i = 10
        'See base class.'
        return self._backend_name

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        if False:
            return 10
        'See base class.'
        msg = f"The node '{self._this_node}' attempts to join the next round of the rendezvous '{self._settings.run_id}'."
        self._record(message=msg)
        log.info(msg)
        try:
            self._stop_heartbeats()
            if self._state_holder.state.round == 0:
                _delay(seconds=(0, 0.3))
            exit_op = _RendezvousExitOp()
            join_op = _RendezvousJoinOp()
            deadline = self._get_deadline(self._settings.timeout.join)
            self._op_executor.run(exit_op, deadline)
            self._op_executor.run(join_op, deadline)
            self._start_heartbeats()
            (rank, world_size) = self._get_world()
            store = self._get_store()
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise
        msg = f"The node '{self._this_node}' has joined round {self._state_holder.state.round} of the rendezvous '{self._settings.run_id}' as rank {rank} in a world of size {world_size}."
        self._record(message=msg, rank=rank)
        log.info(msg)
        return (store, rank, world_size)

    def is_closed(self) -> bool:
        if False:
            while True:
                i = 10
        'See base class.'
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()
                return self._state_holder.state.closed
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def set_closed(self) -> None:
        if False:
            return 10
        'See base class.'
        try:
            with self._heartbeat_lock:
                self._close()
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def num_nodes_waiting(self) -> int:
        if False:
            return 10
        'See base class.'
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()
                return len(self._state_holder.state.wait_list)
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def get_run_id(self) -> str:
        if False:
            while True:
                i = 10
        'See base class.'
        return self._settings.run_id

    def shutdown(self) -> bool:
        if False:
            return 10
        'See base class.'
        self._stop_heartbeats()
        try:
            self._close()
            return True
        except RendezvousError as ex:
            msg = f"The node '{self._this_node}' has failed to shutdown the rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)
            return False
        except Exception as e:
            self._record(message=f'{type(e).__name__}: {str(e)}', node_state=NodeState.FAILED)
            raise

    def _close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        op = _RendezvousCloseOp()
        deadline = self._get_deadline(self._settings.timeout.close)
        self._op_executor.run(op, deadline)
        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.info(msg)

    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self = weak_self()
        if self is not None:
            self._keep_alive()

    def _keep_alive(self) -> None:
        if False:
            i = 10
            return i + 15
        self._heartbeat_lock.acquire()
        op = _RendezvousKeepAliveOp()
        deadline = self._get_deadline(self._settings.timeout.heartbeat)
        try:
            self._op_executor.run(op, deadline)
            msg = f"The node '{self._this_node}' has sent a keep-alive heartbeat to the rendezvous '{self._settings.run_id}'."
            self._record(message=msg)
            log.debug(msg)
        except RendezvousError as ex:
            msg = f"The node '{self._this_node}' has failed to send a keep-alive heartbeat to the rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)
        finally:
            self._heartbeat_lock.release()

    def _start_heartbeats(self) -> None:
        if False:
            i = 10
            return i + 15
        self._keep_alive_timer = _PeriodicTimer(self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self))
        self._keep_alive_timer.set_name(f'RendezvousKeepAliveTimer_{self._this_node.local_id}')
        self._keep_alive_timer.start()

    def _stop_heartbeats(self) -> None:
        if False:
            while True:
                i = 10
        if self._keep_alive_timer is None:
            return
        self._keep_alive_timer.cancel()

    def _get_world(self) -> Tuple[int, int]:
        if False:
            i = 10
            return i + 15
        state = self._state_holder.state
        return (state.participants[self._this_node], len(state.participants))

    def _get_store(self) -> Store:
        if False:
            for i in range(10):
                print('nop')
        key_prefix = f'torch.rendezvous.{self._settings.run_id}.{self._state_holder.state.round}'
        return PrefixStore(key_prefix, self._store)

    def _get_deadline(self, timeout: timedelta) -> float:
        if False:
            i = 10
            return i + 15
        return time.monotonic() + timeout.total_seconds()

def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    if False:
        return 10
    timeout = params.get_as_int(key + '_timeout')
    if timeout is None:
        return None
    return timedelta(seconds=timeout)

def create_handler(store: Store, backend: RendezvousBackend, params: RendezvousParameters) -> DynamicRendezvousHandler:
    if False:
        print('Hello World!')
    'Create a new :py:class:`DynamicRendezvousHandler` from the specified parameters.\n\n    Args:\n        store:\n            The C10d store to return as part of the rendezvous.\n        backend:\n            The backend to use to hold the rendezvous state.\n\n    +-------------------+------------------------------------------------------+\n    | Parameter         | Description                                          |\n    +===================+======================================================+\n    | join_timeout      | The total time, in seconds, within which the         |\n    |                   | rendezvous is expected to complete. Defaults to 600  |\n    |                   | seconds.                                             |\n    +-------------------+------------------------------------------------------+\n    | last_call_timeout | An additional wait amount, in seconds, before        |\n    |                   | completing the rendezvous once the minimum number of |\n    |                   | nodes has been reached. Defaults to 30 seconds.      |\n    +-------------------+------------------------------------------------------+\n    | close_timeout     | The time, in seconds, within which the rendezvous is |\n    |                   | expected to close after a call to                    |\n    |                   | :py:meth:`RendezvousHandler.set_closed` or           |\n    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |\n    |                   | 30 seconds.                                          |\n    +-------------------+------------------------------------------------------+\n    '
    try:
        timeout = RendezvousTimeout(_get_timeout(params, 'join'), _get_timeout(params, 'last_call'), _get_timeout(params, 'close'))
        return DynamicRendezvousHandler.from_backend(params.run_id, store, backend, params.min_nodes, params.max_nodes, params.local_addr, timeout)
    except Exception as e:
        construct_and_record_rdzv_event(message=f'{type(e).__name__}: {str(e)}', run_id=params.run_id, node_state=NodeState.FAILED)
        raise