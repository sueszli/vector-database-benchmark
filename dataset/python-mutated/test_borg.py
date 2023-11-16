import unittest
from patterns.creational.borg import Borg, YourBorg

class BorgTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.b1 = Borg()
        self.b2 = Borg()
        self.ib1 = YourBorg()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.ib1.state = 'Init'

    def test_initial_borg_state_shall_be_init(self):
        if False:
            i = 10
            return i + 15
        b = Borg()
        self.assertEqual(b.state, 'Init')

    def test_changing_instance_attribute_shall_change_borg_state(self):
        if False:
            return 10
        self.b1.state = 'Running'
        self.assertEqual(self.b1.state, 'Running')
        self.assertEqual(self.b2.state, 'Running')
        self.assertEqual(self.ib1.state, 'Running')

    def test_instances_shall_have_own_ids(self):
        if False:
            return 10
        self.assertNotEqual(id(self.b1), id(self.b2), id(self.ib1))