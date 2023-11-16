"""Utils module for dynamic routing game and mean field routing game.

This module has three main classes:
- Network
- Vehicle
- OriginDestinationDemand
"""
from __future__ import annotations
from collections.abc import Collection
from typing import Any, Optional
NO_POSSIBLE_ACTION = 0

def _road_section_from_nodes(origin: str, destination: str) -> str:
    if False:
        while True:
            i = 10
    "Create a road section 'A->B' from two nodes 'A' and 'B'."
    return f'{origin}->{destination}'

def _nodes_from_road_section(movement: str) -> tuple[str, str]:
    if False:
        i = 10
        return i + 15
    "Split a road section 'A->B' to two nodes 'A' and 'B'."
    (origin, destination) = movement.split('->')
    return (origin, destination)

def assign_dictionary_input_to_object(dict_object: dict[str, Any], road_sections: Collection[str], default_value: Any) -> dict[str, Any]:
    if False:
        return 10
    'Check dictionary has road sections has key or return default_value dict.'
    if dict_object:
        assert set(dict_object) == set(road_sections), 'Objects are not defined for each road sections.'
        return dict_object
    dict_object_returned = {}
    for road_section in road_sections:
        dict_object_returned[road_section] = default_value
    return dict_object_returned

class Network:
    """Network implementation.

  A network is basically a directed graph with a volume delay function on each
  of its edges. Each vertex is refered to as a string (for example "A") and each
  edge as a string f"{node1}->{node2}" (for example "A->B"). The network is
  created from a adjacency list. Each road section is mapped to an action index
  (positive integer) in _action_by_road_section. The volume delay function on
  each road section rs is given by
  _free_flow_travel_time[rs]*(1+ _a[rs]*(v/_capacity[rs])**_b[rs])
  where v is the volume on the road section rs, according to the U.S. Bureau of
  Public Road (BPR). Such functions are called fundamental diagram of traffic
  flow.

  If one would like to plot the network then node position should be passed
  in the constructor. Then return_list_for_matplotlib_quiver can be used with
  Matplotlib:
  ```python3
  fig, ax = plt.subplots()
  o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
  ax.quiver(o_xs, o_ys, np.subtract(d_xs, o_xs), np.subtract(d_ys, o_ys),
            color="b", angles='xy', scale_units='xy', scale=1)
  ```

  See the Network tests for an example.
  Attributes: _a, _b, _capacity, _free_flow_travel_time: dictionary that maps
  road section string representation to its a, b, relative capacity and free
  flow travel time coefficient in its BPR function.
    _action_by_road_section: dictionary that maps road section to action id.
    _adjacency_list: adjacency list of the line graph of the road network.
    _node_position: dictionary that maps node to couple of float encoding x and
    y position of the node. None by default.
    _road_section_by_action: dictionary that maps action id to road section.
  """
    _a: dict[str, float]
    _b: dict[str, float]
    _action_by_road_section: dict[str, int]
    _adjacency_list: dict[str, Collection[str]]
    _capacity: dict[str, float]
    _free_flow_travel_time: dict[str, float]
    _node_position: dict[str, tuple[float, float]]
    _road_section_by_action: dict[int, str]

    def __init__(self, adjacency_list: dict[str, Collection[str]], node_position: Optional[dict[str, tuple[float, float]]]=None, bpr_a_coefficient: Optional[dict[str, float]]=None, bpr_b_coefficient: Optional[dict[str, float]]=None, capacity: Optional[dict[str, float]]=None, free_flow_travel_time: Optional[dict[str, float]]=None):
        if False:
            while True:
                i = 10
        self._adjacency_list = adjacency_list
        self._action_by_road_section = self._create_action_by_road_section()
        self._road_section_by_action = {v: k for (k, v) in self._action_by_road_section.items()}
        nodes = set(adjacency_list)
        assert all((destination_node in nodes for destination_nodes in self._adjacency_list.values() for destination_node in destination_nodes)), 'Adjacency list is not correct.'
        if node_position:
            assert set(node_position) == nodes
            self._node_position = node_position
        else:
            self._node_position = None
        self._a = assign_dictionary_input_to_object(bpr_a_coefficient, self._action_by_road_section, 0)
        self._b = assign_dictionary_input_to_object(bpr_b_coefficient, self._action_by_road_section, 1)
        self._capacity = assign_dictionary_input_to_object(capacity, self._action_by_road_section, 1)
        self._free_flow_travel_time = assign_dictionary_input_to_object(free_flow_travel_time, self._action_by_road_section, 1)
        assert hasattr(self, '_adjacency_list')
        assert hasattr(self, '_node_position')
        assert hasattr(self, '_a')
        assert hasattr(self, '_b')
        assert hasattr(self, '_capacity')
        assert hasattr(self, '_free_flow_travel_time')

    def _create_action_by_road_section(self) -> tuple[set[str], dict[int, str]]:
        if False:
            return 10
        'Create dictionary that maps movement to action.\n\n    The dictionary that maps movement to action is used to define the action\n    from a movement that a vehicle would like to do.\n    Returns:\n      action_by_road_section: dictionary with key begin a movement for example\n        "O->A" and value the action numbers. Action numbers are succesive\n        integers indexed from 1.\n    '
        action_by_road_section = {}
        action_number = NO_POSSIBLE_ACTION + 1
        for (origin, successors) in sorted(self._adjacency_list.items()):
            for destination in successors:
                road_section = _road_section_from_nodes(origin, destination)
                if road_section in action_by_road_section:
                    raise ValueError(f'{road_section} exists twice in the adjacency list. The current network implementation does not enable parallel links.')
                action_by_road_section[road_section] = action_number
                action_number += 1
        return action_by_road_section

    def num_links(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the number of road sections.'
        return len(self._action_by_road_section)

    def num_actions(self) -> int:
        if False:
            return 10
        'Returns the number of possible actions.\n\n    Equal to the number of road section + 1. An action could either be moving to\n    a specific road section or not move.\n    '
        return 1 + self.num_links()

    def links(self) -> list[str]:
        if False:
            while True:
                i = 10
        'Returns the road sections as a list.'
        return list(self._action_by_road_section)

    def get_successors(self, node: str) -> Collection[str]:
        if False:
            return 10
        'Returns the successor nodes of the node.'
        return self._adjacency_list[node]

    def get_action_id_from_movement(self, origin: str, destination: str) -> int:
        if False:
            while True:
                i = 10
        'Maps two connected nodes to an action.'
        return self._action_by_road_section[_road_section_from_nodes(origin, destination)]

    def get_road_section_from_action_id(self, action_id: int) -> str:
        if False:
            return 10
        'Maps a action to the corresponding road section.'
        return self._road_section_by_action[action_id]

    def is_location_at_sink_node(self, road_section: str) -> bool:
        if False:
            print('Hello World!')
        'Returns True if the road section has no successors.'
        (start_section, end_section_node) = _nodes_from_road_section(road_section)
        if start_section not in self._adjacency_list:
            raise KeyError(f'{start_section} is not a network node.')
        return not self.get_successors(end_section_node)

    def check_list_of_vehicles_is_correct(self, vehicles: Collection['Vehicle']):
        if False:
            for i in range(10):
                print('nop')
        'Assert that vehicles have valid origin and destination.'
        for vehicle in vehicles:
            if vehicle.origin not in self._action_by_road_section or vehicle.destination not in self._action_by_road_section:
                raise ValueError(f'Incorrect origin or destination for {vehicle}')

    def check_list_of_od_demand_is_correct(self, vehicles: Collection['OriginDestinationDemand']):
        if False:
            print('Hello World!')
        'Assert that OD demands have valid origin and destination.'
        for vehicle in vehicles:
            if vehicle.origin not in self._action_by_road_section or vehicle.destination not in self._action_by_road_section:
                raise ValueError(f'Incorrect origin or destination for {vehicle}')

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return str(self._adjacency_list)

    def get_travel_time(self, road_section: str, volume: float) -> int:
        if False:
            i = 10
            return i + 15
        'Returns travel time on the road section given the volume on it.\n\n    Volume unit should be the same as the capacity unit.\n    Travel time unit is the free flow travel time unit.\n    Args:\n      road_section: the road section.\n      volume: the volume on the road section.\n    '
        return self._free_flow_travel_time[road_section] * (1.0 + self._a[road_section] * (volume / self._capacity[road_section]) ** self._b[road_section])

    def assert_valid_action(self, action: int, road_section: str=None):
        if False:
            i = 10
            return i + 15
        'Assert that an action as a int is valid.\n\n    The action should be a int between 1 and num_actions. In case road_section\n    is not None then it is test if the action correspond to going on a road\n    section which is a successor of road_section.\n\n    Args:\n      action: the action,\n      road_section: the road section.\n    '
        assert isinstance(action, int), f'{action} is not a int.'
        assert 1 <= action < self.num_actions(), str(action)
        if road_section is not None:
            new_road_section = self.get_road_section_from_action_id(action)
            (origin_new_section, end_new_section) = _nodes_from_road_section(new_road_section)
            (_, end_section_node) = _nodes_from_road_section(road_section)
            assert end_section_node == origin_new_section, f'The action is not legal, trying to go to {new_road_section} from {road_section} without going through {end_section_node}.'
            successors = self.get_successors(origin_new_section)
            assert end_new_section in successors, f'Invalid action {new_road_section}. It is not a successors of {end_section_node}: {successors}.'

    def return_position_of_road_section(self, road_section: str) -> tuple[float, float]:
        if False:
            print('Hello World!')
        'Returns position of the middle of theroad section as (x,y).'
        assert self._node_position is not None, 'The network should have node positions in order to be plot.'
        (o_link, d_link) = _nodes_from_road_section(road_section)
        (o_x, o_y) = self._node_position[o_link]
        (d_x, d_y) = self._node_position[d_link]
        return ((o_x + d_x) / 2, (o_y + d_y) / 2)

    def return_list_for_matplotlib_quiver(self) -> tuple[list[float], list[float], list[float], list[float]]:
        if False:
            return 10
        'Returns 4 list of encoding the positions of the road sections.\n\n    ```python3\n    fig, ax = plt.subplots()\n    o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()\n    ax.quiver(o_xs, o_ys, np.subtract(d_xs, o_xs), np.subtract(d_ys, o_ys),\n              color="b", angles=\'xy\', scale_units=\'xy\', scale=1)\n    ```\n    will show the network.\n    Returns:\n      o_xs, o_ys, d_xs, d_ys: list of the start x and y positions and of the end\n        x and y postions of each road section. Each element of each list\n        corresponds to one road section.\n    '
        assert self._node_position is not None, 'The network should have node positions in order to be plot.'
        o_xs = []
        o_ys = []
        d_xs = []
        d_ys = []
        for road_section in self._action_by_road_section:
            (o_link, d_link) = _nodes_from_road_section(road_section)
            (o_x, o_y) = self._node_position[o_link]
            (d_x, d_y) = self._node_position[d_link]
            o_xs.append(o_x)
            o_ys.append(o_y)
            d_xs.append(d_x)
            d_ys.append(d_y)
        return (o_xs, o_ys, d_xs, d_ys)

class Vehicle:
    """A Vehicle is one origin and one destination.

  Both the origin and the destination of the vehicle are road section, therefore
  they are string formatted as "{str}->{str}".
  Attributes:
    destination: destination of the vehicle.
    origin: origin of the vehicle.
    departure_time: departure time of the vehicle.
  """
    _destination: str
    _origin: str
    _departure_time: float

    def __init__(self, origin: str, destination: str, departure_time: float=0.0):
        if False:
            while True:
                i = 10
        assert all(('->' in node for node in [origin, destination]))
        self._origin = origin
        self._destination = destination
        self._departure_time = departure_time

    @property
    def origin(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Returns vehicle's origin."
        return self._origin

    @property
    def destination(self) -> str:
        if False:
            return 10
        "Returns vehicle's destination."
        return self._destination

    @property
    def departure_time(self) -> float:
        if False:
            print('Hello World!')
        "Returns vehicle's departure time."
        return self._departure_time

    def __str__(self):
        if False:
            return 10
        return f'Vehicle with origin {self.origin}, destination {self.destination} and departure time {self._departure_time}.'

class OriginDestinationDemand(Vehicle):
    """Number of trips from origin to destination for a specific departure time.

  Both the origin and the destination of the vehicle are road section, therefore
  they are string formatted as "{str}->{str}".
  Attributes:
    destination: destination of the vehicles.
    origin: origin of the vehicles.
    departure_time: departure time of the vehicles.
    counts: the number of vehicles with the origin, destination and departure
      time.
  """
    _counts: float

    def __init__(self, origin: str, destination: str, departure_time: float, counts: float):
        if False:
            while True:
                i = 10
        super().__init__(origin, destination, departure_time)
        self._counts = counts

    @property
    def counts(self) -> float:
        if False:
            print('Hello World!')
        'Returns the number of vehicles in the instance.'
        return self._counts

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{self._counts} with origin {self.origin}, destination {self.destination} and departure time {self._departure_time}.'