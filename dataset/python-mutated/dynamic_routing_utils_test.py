"""Tests for Python dynamic routing game utils."""
from absl.testing import absltest
from open_spiel.python.games import dynamic_routing_utils as utils

class NetworkTest(absltest.TestCase):
    """Tests for Network class."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Create a network O->A->D for testing.'
        super().setUp()
        self.network = utils.Network({'O': ['A'], 'A': ['D'], 'D': []})

    def test_adjacency_list_init(self):
        if False:
            i = 10
            return i + 15
        'Test class instanciation with adjacency list.'
        self.assertEqual(self.network.num_links(), 2)
        self.assertEqual(self.network.get_successors('O'), ['A'])
        self.assertEqual(self.network.get_successors('A'), ['D'])
        self.assertEqual(self.network.get_successors('D'), [])
        self.assertTrue(self.network.is_location_at_sink_node('A->D'))
        self.assertFalse(self.network.is_location_at_sink_node('O->A'))
        self.assertEqual(self.network.get_action_id_from_movement('A', 'D'), 1)
        self.assertEqual(self.network.get_action_id_from_movement('O', 'A'), 2)
        self.assertEqual(self.network.get_road_section_from_action_id(1), 'A->D')
        self.assertEqual(self.network.get_road_section_from_action_id(2), 'O->A')

    def test_get_successors_with_wrong_node(self):
        if False:
            print('Hello World!')
        'Test get successors on non existing node.'
        with self.assertRaises(KeyError):
            self.network.get_successors('Z')

    def test_get_action_id_without_connected_nodes(self):
        if False:
            i = 10
            return i + 15
        'Test get actions id on non connected nodes.'
        with self.assertRaises(KeyError):
            self.network.get_action_id_from_movement('O', 'D')

    def test_get_action_id_with_wrong_nodes(self):
        if False:
            i = 10
            return i + 15
        'Test get actions id on non existing node.'
        with self.assertRaises(KeyError):
            self.network.get_action_id_from_movement('Z', 'D')

    def test_is_location_at_sink_noded_with_wrong_road_section(self):
        if False:
            for i in range(10):
                print('nop')
        'Test is_location_at_sink_node on non existing second node.'
        with self.assertRaises(KeyError):
            self.network.is_location_at_sink_node('A->Z')

    def test_is_location_at_sink_noded_with_wrong_road_section_2(self):
        if False:
            while True:
                i = 10
        'Test is_location_at_sink_node on non existing first node.'
        with self.assertRaises(KeyError):
            self.network.is_location_at_sink_node('Z->D')

    def test_is_location_at_sink_noded_with_wrong_arg(self):
        if False:
            i = 10
            return i + 15
        'Test is_location_at_sink_node on wrong link str representation.'
        with self.assertRaises(ValueError):
            self.network.is_location_at_sink_node('D')

    def test_get_road_section_with_action_id(self):
        if False:
            while True:
                i = 10
        'Test get_road_section_from_action_id on non possible action.'
        with self.assertRaises(KeyError):
            self.network.get_road_section_from_action_id(0)

    def test_num_links_method(self):
        if False:
            print('Hello World!')
        pass

    def test_num_actions_method(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_links(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_list_of_vehicles_is_correct_method(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_list_of_od_demand_is_correct_method(self):
        if False:
            print('Hello World!')
        pass

    def test_str_method(self):
        if False:
            while True:
                i = 10
        pass

    def test_get_travel_time_methods(self):
        if False:
            print('Hello World!')
        pass

    def test_assert_valid_action_methods(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_default_travel_time_methods(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_customable_travel_time_methods(self):
        if False:
            i = 10
            return i + 15
        pass

class VehicleTest(absltest.TestCase):
    """Tests for Vehicle class."""

    def test_vehicle_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test instanciation of Vehicle.'
        vehicle = utils.Vehicle('O->A', 'B->D')
        self.assertEqual(vehicle.destination, 'B->D')
        self.assertEqual(vehicle.origin, 'O->A')
        self.assertEqual(vehicle.departure_time, 0)

    def test_vehicle_2(self):
        if False:
            print('Hello World!')
        'Test instanciation of with departure time.'
        vehicle = utils.Vehicle('O->A', 'B->D', 10.5)
        self.assertEqual(vehicle.origin, 'O->A')
        self.assertEqual(vehicle.destination, 'B->D')
        self.assertEqual(vehicle.departure_time, 10.5)

class OriginDestinationDemandTest(absltest.TestCase):
    """Tests for OriginDestinationDemand class."""

    def test_od_demand_1(self):
        if False:
            print('Hello World!')
        'Test instanciation of OD demand.'
        od_demand = utils.OriginDestinationDemand('O->A', 'B->D', 0, 30)
        self.assertEqual(od_demand.destination, 'B->D')
        self.assertEqual(od_demand.origin, 'O->A')
        self.assertEqual(od_demand.departure_time, 0)
        self.assertEqual(od_demand.counts, 30)

    def test_od_demand_2(self):
        if False:
            while True:
                i = 10
        'Test instanciation of OD demand.'
        od_demand = utils.OriginDestinationDemand('O->A', 'B->D', 10.5, 43.2)
        self.assertEqual(od_demand.origin, 'O->A')
        self.assertEqual(od_demand.destination, 'B->D')
        self.assertEqual(od_demand.departure_time, 10.5)
        self.assertEqual(od_demand.counts, 43.2)
if __name__ == '__main__':
    absltest.main()