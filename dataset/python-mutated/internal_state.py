from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Iterable, Collection
from pyflink.datastream.state import State, ValueState, AppendingState, MergingState, ListState, AggregatingState, ReducingState, MapState, ReadOnlyBroadcastState, BroadcastState
N = TypeVar('N')
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
IN = TypeVar('IN')
OUT = TypeVar('OUT')

class InternalKvState(State, Generic[N]):
    """
    The :class:InternalKvState is the root of the internal state type hierarchy, similar to the
    :class:State being the root of the public API state hierarchy.

    The internal state classes give access to the namespace getters and setters and access to
    additional functionality, like raw value access or state merging.

    The public API state hierarchy is intended to be programmed against by Flink applications. The
    internal state hierarchy holds all the auxiliary methods that are used by the runtime and not
    intended to be used by user applications. These internal methods are considered of limited use
    to users and only confusing, and are usually not regarded as stable across releases.

    """

    @abstractmethod
    def set_current_namespace(self, namespace: N) -> None:
        if False:
            print('Hello World!')
        '\n        Sets the current namespace, which will be used when using the state access methods.\n\n        :param namespace: The namespace.\n        '
        pass

class InternalValueState(InternalKvState[N], ValueState[T], ABC):
    """
    The peer to the :class:ValueState in the internal state type hierarchy.
    """
    pass

class InternalAppendingState(InternalKvState[N], AppendingState[IN, OUT], ABC):
    """
    The peer to the :class:AppendingState in the internal state type hierarchy.
    """
    pass

class InternalMergingState(InternalAppendingState[N, IN, OUT], MergingState[IN, OUT]):
    """
    The peer to the :class:MergingState in the internal state type hierarchy.
    """

    @abstractmethod
    def merge_namespaces(self, target: N, sources: Collection[N]) -> None:
        if False:
            print('Hello World!')
        '\n        Merges the state of the current key for the given source namespaces into the state of the\n        target namespace.\n\n        :param target: The target namespace where the merged state should be stored.\n        :param sources: The source namespaces whose state should be merged.\n        '
        pass

class InternalListState(InternalMergingState[N, List[T], Iterable[T]], ListState[T], ABC):
    """
    The peer to the :class:ListState in the internal state type hierarchy.
    """
    pass

class InternalAggregatingState(InternalMergingState[N, IN, OUT], AggregatingState[IN, OUT], ABC):
    """
    The peer to the :class:AggregatingState in the internal state type hierarchy.
    """
    pass

class InternalReducingState(InternalMergingState[N, T, T], ReducingState[T], ABC):
    """
    The peer to the :class:ReducingState in the internal state type hierarchy.
    """
    pass

class InternalMapState(InternalKvState[N], MapState[K, V], ABC):
    """
    The peer to the :class:MapState in the internal state type hierarchy.
    """
    pass

class InternalReadOnlyBroadcastState(ReadOnlyBroadcastState[K, V], ABC):
    """
    The peer to :class:`ReadOnlyBroadcastState`.
    """
    pass

class InternalBroadcastState(InternalReadOnlyBroadcastState[K, V], BroadcastState[K, V], ABC):
    """
    The peer to :class:`BroadcastState`.
    """

    @abstractmethod
    def to_read_only_broadcast_state(self) -> InternalReadOnlyBroadcastState[K, V]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert to :class:`ReadOnlyBroadcastState` interface with the same underlying state.\n        '
        pass