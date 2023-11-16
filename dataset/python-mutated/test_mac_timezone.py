"""
Integration tests for mac_timezone

If using parallels, make sure Time sync is turned off. Otherwise, parallels will
keep changing your date/time settings while the tests are running. To turn off
Time sync do the following:
    - Go to actions -> configure
    - Select options at the top and 'More Options' on the left
    - Set time to 'Do not sync'
"""
import datetime
import pytest
from tests.support.case import ModuleCase

@pytest.mark.flaky(max_runs=4)
@pytest.mark.skip_unless_on_darwin
@pytest.mark.skip_if_binaries_missing('systemsetup')
@pytest.mark.skip_if_not_root
@pytest.mark.slow_test
class MacTimezoneModuleTest(ModuleCase):
    """
    Validate the mac_timezone module
    """
    USE_NETWORK_TIME = False
    TIME_SERVER = 'time.apple.com'
    TIME_ZONE = ''
    CURRENT_DATE = ''
    CURRENT_TIME = ''

    def setUp(self):
        if False:
            return 10
        '\n        Get current settings\n        '
        self.USE_NETWORK_TIME = self.run_function('timezone.get_using_network_time')
        self.TIME_SERVER = self.run_function('timezone.get_time_server')
        self.TIME_ZONE = self.run_function('timezone.get_zone')
        self.CURRENT_DATE = self.run_function('timezone.get_date')
        self.CURRENT_TIME = self.run_function('timezone.get_time')
        self.run_function('timezone.set_using_network_time', [False])
        self.run_function('timezone.set_zone', ['America/Denver'])

    def tearDown(self):
        if False:
            print('Hello World!')
        '\n        Reset to original settings\n        '
        self.run_function('timezone.set_time_server', [self.TIME_SERVER])
        self.run_function('timezone.set_using_network_time', [self.USE_NETWORK_TIME])
        self.run_function('timezone.set_zone', [self.TIME_ZONE])
        if not self.USE_NETWORK_TIME:
            self.run_function('timezone.set_date', [self.CURRENT_DATE])
            self.run_function('timezone.set_time', [self.CURRENT_TIME])

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_get_set_date(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test timezone.get_date\n        Test timezone.set_date\n        '
        self.assertTrue(self.run_function('timezone.set_date', ['2/20/2011']))
        self.assertEqual(self.run_function('timezone.get_date'), '2/20/2011')
        self.assertEqual(self.run_function('timezone.set_date', ['13/12/2014']), "ERROR executing 'timezone.set_date': Invalid Date/Time Format: 13/12/2014")

    @pytest.mark.slow_test
    def test_get_time(self):
        if False:
            i = 10
            return i + 15
        '\n        Test timezone.get_time\n        '
        text_time = self.run_function('timezone.get_time')
        self.assertNotEqual(text_time, 'Invalid Timestamp')
        obj_date = datetime.datetime.strptime(text_time, '%H:%M:%S')
        self.assertIsInstance(obj_date, datetime.date)

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_set_time(self):
        if False:
            while True:
                i = 10
        '\n        Test timezone.set_time\n        '
        self.assertTrue(self.run_function('timezone.set_time', ['3:14']))
        self.assertEqual(self.run_function('timezone.set_time', ['3:71']), "ERROR executing 'timezone.set_time': Invalid Date/Time Format: 3:71")

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_get_set_zone(self):
        if False:
            i = 10
            return i + 15
        '\n        Test timezone.get_zone\n        Test timezone.set_zone\n        '
        self.assertTrue(self.run_function('timezone.set_zone', ['Pacific/Wake']))
        self.assertEqual(self.run_function('timezone.get_zone'), 'Pacific/Wake')
        self.assertEqual(self.run_function('timezone.set_zone', ['spongebob']), "ERROR executing 'timezone.set_zone': Invalid Timezone: spongebob")

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_get_offset(self):
        if False:
            print('Hello World!')
        '\n        Test timezone.get_offset\n        '
        self.assertTrue(self.run_function('timezone.set_zone', ['Pacific/Wake']))
        self.assertIsInstance(self.run_function('timezone.get_offset'), (str,))
        self.assertEqual(self.run_function('timezone.get_offset'), '+1200')
        self.assertTrue(self.run_function('timezone.set_zone', ['America/Los_Angeles']))
        self.assertIsInstance(self.run_function('timezone.get_offset'), (str,))
        self.assertEqual(self.run_function('timezone.get_offset'), '-0700')

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_get_set_zonecode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test timezone.get_zonecode\n        Test timezone.set_zonecode\n        '
        self.assertTrue(self.run_function('timezone.set_zone', ['America/Los_Angeles']))
        self.assertIsInstance(self.run_function('timezone.get_zonecode'), (str,))
        self.assertEqual(self.run_function('timezone.get_zonecode'), 'PDT')
        self.assertTrue(self.run_function('timezone.set_zone', ['Pacific/Wake']))
        self.assertIsInstance(self.run_function('timezone.get_zonecode'), (str,))
        self.assertEqual(self.run_function('timezone.get_zonecode'), 'WAKT')

    @pytest.mark.slow_test
    def test_list_zones(self):
        if False:
            i = 10
            return i + 15
        '\n        Test timezone.list_zones\n        '
        zones = self.run_function('timezone.list_zones')
        self.assertIsInstance(self.run_function('timezone.list_zones'), list)
        self.assertIn('America/Denver', self.run_function('timezone.list_zones'))
        self.assertIn('America/Los_Angeles', self.run_function('timezone.list_zones'))

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_zone_compare(self):
        if False:
            i = 10
            return i + 15
        '\n        Test timezone.zone_compare\n        '
        self.assertTrue(self.run_function('timezone.set_zone', ['America/Denver']))
        self.assertTrue(self.run_function('timezone.zone_compare', ['America/Denver']))
        self.assertFalse(self.run_function('timezone.zone_compare', ['Pacific/Wake']))

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_get_set_using_network_time(self):
        if False:
            return 10
        '\n        Test timezone.get_using_network_time\n        Test timezone.set_using_network_time\n        '
        self.assertTrue(self.run_function('timezone.set_using_network_time', [True]))
        self.assertTrue(self.run_function('timezone.get_using_network_time'))
        self.assertTrue(self.run_function('timezone.set_using_network_time', [False]))
        self.assertFalse(self.run_function('timezone.get_using_network_time'))

    @pytest.mark.skip(reason='Skip until we can figure out why modifying the system clock causes ZMQ errors')
    @pytest.mark.destructive_test
    def test_get_set_time_server(self):
        if False:
            i = 10
            return i + 15
        '\n        Test timezone.get_time_server\n        Test timezone.set_time_server\n        '
        self.assertTrue(self.run_function('timezone.set_time_server', ['spongebob.com']))
        self.assertEqual(self.run_function('timezone.get_time_server'), 'spongebob.com')