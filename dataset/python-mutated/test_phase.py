import unittest
from odoo.tests import common
test_state = None

def setUpModule():
    if False:
        return 10
    global test_state
    test_state = {}

def tearDownModule():
    if False:
        while True:
            i = 10
    global test_state
    test_state = None

class TestPhaseInstall00(unittest.TestCase):
    """
    WARNING: Relies on tests being run in alphabetical order
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.state = None

    def test_00_setup(self):
        if False:
            for i in range(10):
                print('nop')
        type(self).state = 'init'

    @common.at_install(False)
    def test_01_no_install(self):
        if False:
            print('Hello World!')
        type(self).state = 'error'

    def test_02_check(self):
        if False:
            return 10
        self.assertEqual(self.state, 'init', 'Testcase state should not have been transitioned from 00')

class TestPhaseInstall01(unittest.TestCase):
    at_install = False

    def test_default_norun(self):
        if False:
            for i in range(10):
                print('nop')
        self.fail('An unmarket test in a non-at-install case should not run')

    @common.at_install(True)
    def test_set_run(self):
        if False:
            for i in range(10):
                print('nop')
        test_state['set_at_install'] = True

class TestPhaseInstall02(unittest.TestCase):
    """
    Can't put the check for test_set_run in the same class: if
    @common.at_install does not work for test_set_run, it won't work for
    the other one either. Thus move checking of whether test_set_run has
    correctly run indeed to a separate class.

    Warning: relies on *classes* being run in alphabetical order in test
    modules
    """

    def test_check_state(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(test_state.get('set_at_install'), 'The flag should be set if local overriding of runstate')