from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union
from pyflink.datastream.state import ListState, MapState
from pyflink.fn_execution.coders import from_proto, PickleCoder
from pyflink.fn_execution.internal_state import InternalListState, InternalMapState
from pyflink.fn_execution.utils.operation_utils import is_built_in_function, load_aggregate_function
from pyflink.table import FunctionContext
from pyflink.table.data_view import ListView, MapView, DataView

def extract_data_view_specs_from_accumulator(current_index, accumulator):
    if False:
        i = 10
        return i + 15
    i = -1
    extracted_specs = []
    for field in accumulator:
        i += 1
        if isinstance(field, MapView):
            extracted_specs.append(MapViewSpec('builtInAgg%df%d' % (current_index, i), i, PickleCoder(), PickleCoder()))
        elif isinstance(field, ListView):
            extracted_specs.append(ListViewSpec('builtInAgg%df%d' % (current_index, i), i, PickleCoder()))
    return extracted_specs

def extract_data_view_specs(udfs):
    if False:
        while True:
            i = 10
    extracted_udf_data_view_specs = []
    current_index = -1
    for udf in udfs:
        current_index += 1
        udf_data_view_specs_proto = udf.specs
        if not udf_data_view_specs_proto:
            if is_built_in_function(udf.payload):
                built_in_function = load_aggregate_function(udf.payload)
                accumulator = built_in_function.create_accumulator()
                extracted_udf_data_view_specs.append(extract_data_view_specs_from_accumulator(current_index, accumulator))
            else:
                extracted_udf_data_view_specs.append([])
        else:
            extracted_specs = []
            for spec_proto in udf_data_view_specs_proto:
                state_id = spec_proto.name
                field_index = spec_proto.field_index
                if spec_proto.HasField('list_view'):
                    element_coder = from_proto(spec_proto.list_view.element_type)
                    extracted_specs.append(ListViewSpec(state_id, field_index, element_coder))
                elif spec_proto.HasField('map_view'):
                    key_coder = from_proto(spec_proto.map_view.key_type)
                    value_coder = from_proto(spec_proto.map_view.value_type)
                    extracted_specs.append(MapViewSpec(state_id, field_index, key_coder, value_coder))
                else:
                    raise Exception('Unsupported data view spec type: ' + spec_proto.type)
            extracted_udf_data_view_specs.append(extracted_specs)
    if all([len(i) == 0 for i in extracted_udf_data_view_specs]):
        return []
    return extracted_udf_data_view_specs
N = TypeVar('N')

class StateDataView(DataView, Generic[N]):

    @abstractmethod
    def set_current_namespace(self, namespace: N):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets current namespace for state.\n        '
        pass

class StateListView(ListView, StateDataView[N], ABC):

    def __init__(self, list_state: Union[ListState, InternalListState]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._list_state = list_state

    def get(self):
        if False:
            return 10
        return self._list_state.get()

    def add(self, value):
        if False:
            i = 10
            return i + 15
        self._list_state.add(value)

    def add_all(self, values):
        if False:
            i = 10
            return i + 15
        self._list_state.add_all(values)

    def clear(self):
        if False:
            return 10
        self._list_state.clear()

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash([i for i in self.get()])

class KeyedStateListView(StateListView[N]):
    """
    KeyedStateListView is a default implementation of StateListView whose underlying
    representation is a keyed state.
    """

    def __init__(self, list_state: ListState):
        if False:
            return 10
        super(KeyedStateListView, self).__init__(list_state)

    def set_current_namespace(self, namespace: N):
        if False:
            i = 10
            return i + 15
        raise Exception("KeyedStateListView doesn't support set_current_namespace")

class NamespacedStateListView(StateListView[N]):
    """
    NamespacedStateListView is a StateListView whose underlying representation is a keyed and
    namespaced state. It also supports changing current namespace.
    """

    def __init__(self, list_state: InternalListState):
        if False:
            return 10
        super(NamespacedStateListView, self).__init__(list_state)

    def set_current_namespace(self, namespace: N):
        if False:
            for i in range(10):
                print('nop')
        self._list_state.set_current_namespace(namespace)

class StateMapView(MapView, StateDataView[N], ABC):

    def __init__(self, map_state: Union[MapState, InternalMapState]):
        if False:
            return 10
        super().__init__()
        self._map_state = map_state

    def get(self, key):
        if False:
            return 10
        return self._map_state.get(key)

    def put(self, key, value) -> None:
        if False:
            i = 10
            return i + 15
        self._map_state.put(key, value)

    def put_all(self, dict_value) -> None:
        if False:
            return 10
        self._map_state.put_all(dict_value)

    def remove(self, key) -> None:
        if False:
            while True:
                i = 10
        self._map_state.remove(key)

    def contains(self, key) -> bool:
        if False:
            print('Hello World!')
        return self._map_state.contains(key)

    def items(self):
        if False:
            i = 10
            return i + 15
        return self._map_state.items()

    def keys(self):
        if False:
            while True:
                i = 10
        return self._map_state.keys()

    def values(self):
        if False:
            while True:
                i = 10
        return self._map_state.values()

    def is_empty(self) -> bool:
        if False:
            while True:
                i = 10
        return self._map_state.is_empty()

    def clear(self) -> None:
        if False:
            print('Hello World!')
        return self._map_state.clear()

class KeyedStateMapView(StateMapView[N]):
    """
    KeyedStateMapView is a default implementation of StateMapView whose underlying
    representation is a keyed state.
    """

    def __init__(self, map_state: MapState):
        if False:
            while True:
                i = 10
        super(KeyedStateMapView, self).__init__(map_state)

    def set_current_namespace(self, namespace: N):
        if False:
            print('Hello World!')
        raise Exception("KeyedStateMapView doesn't support set_current_namespace")

class NamespacedStateMapView(StateMapView[N]):
    """
    NamespacedStateMapView is a StateMapView whose underlying representation is a keyed and
    namespaced state. It also supports changing current namespace.
    """

    def __init__(self, map_state: InternalMapState):
        if False:
            for i in range(10):
                print('nop')
        super(NamespacedStateMapView, self).__init__(map_state)

    def set_current_namespace(self, namespace: N):
        if False:
            print('Hello World!')
        self._map_state.set_current_namespace(namespace)

class DataViewSpec(object):

    def __init__(self, state_id, field_index):
        if False:
            for i in range(10):
                print('nop')
        self.state_id = state_id
        self.field_index = field_index

class ListViewSpec(DataViewSpec):

    def __init__(self, state_id, field_index, element_coder):
        if False:
            return 10
        super(ListViewSpec, self).__init__(state_id, field_index)
        self.element_coder = element_coder

class MapViewSpec(DataViewSpec):

    def __init__(self, state_id, field_index, key_coder, value_coder):
        if False:
            while True:
                i = 10
        super(MapViewSpec, self).__init__(state_id, field_index)
        self.key_coder = key_coder
        self.value_coder = value_coder

class StateDataViewStore(ABC):
    """
    This interface contains methods for registering StateDataView with a managed store.
    """

    def __init__(self, function_context: FunctionContext, keyed_state_backend):
        if False:
            return 10
        self._function_context = function_context
        self._keyed_state_backend = keyed_state_backend

    def get_runtime_context(self):
        if False:
            print('Hello World!')
        return self._function_context

    @abstractmethod
    def get_state_list_view(self, state_name, element_coder):
        if False:
            return 10
        '\n        Creates a state list view.\n\n        :param state_name: The name of underlying state of the list view.\n        :param element_coder: The element coder\n        :return: a keyed list state\n        '
        pass

    @abstractmethod
    def get_state_map_view(self, state_name, key_coder, value_coder):
        if False:
            while True:
                i = 10
        '\n        Creates a state map view.\n\n        :param state_name: The name of underlying state of the map view.\n        :param key_coder: The key coder\n        :param value_coder: The value coder\n        :return: a keyed map state\n        '
        pass

class PerKeyStateDataViewStore(StateDataViewStore):
    """
    Default implementation of StateDataViewStore.
    """

    def __init__(self, function_context: FunctionContext, keyed_state_backend):
        if False:
            for i in range(10):
                print('nop')
        super(PerKeyStateDataViewStore, self).__init__(function_context, keyed_state_backend)

    def get_state_list_view(self, state_name, element_coder):
        if False:
            for i in range(10):
                print('nop')
        return KeyedStateListView(self._keyed_state_backend.get_list_state(state_name, element_coder))

    def get_state_map_view(self, state_name, key_coder, value_coder):
        if False:
            return 10
        return KeyedStateMapView(self._keyed_state_backend.get_map_state(state_name, key_coder, value_coder))

class PerWindowStateDataViewStore(StateDataViewStore):
    """
    An implementation of StateDataViewStore for window aggregates which forwards the state
    registration to an underlying RemoteKeyedStateBackend. The state created by this store has the
    ability to switch window namespaces.
    """

    def __init__(self, function_context: FunctionContext, keyed_state_backend):
        if False:
            i = 10
            return i + 15
        super(PerWindowStateDataViewStore, self).__init__(function_context, keyed_state_backend)

    def get_state_list_view(self, state_name, element_coder):
        if False:
            for i in range(10):
                print('nop')
        return NamespacedStateListView(self._keyed_state_backend.get_list_state(state_name, element_coder))

    def get_state_map_view(self, state_name, key_coder, value_coder):
        if False:
            print('Hello World!')
        return NamespacedStateMapView(self._keyed_state_backend.get_map_state(state_name, key_coder, value_coder))