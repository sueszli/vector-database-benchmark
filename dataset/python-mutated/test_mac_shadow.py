"""
integration tests for mac_shadow
"""
import datetime
import pytest
from saltfactories.utils import random_string
from tests.support.case import ModuleCase
TEST_USER = random_string('RS-', lowercase=False)
NO_USER = random_string('RS-', lowercase=False)

@pytest.mark.skip_if_binaries_missing('dscl', 'pwpolicy')
@pytest.mark.skip_if_not_root
@pytest.mark.skip_unless_on_darwin
class MacShadowModuleTest(ModuleCase):
    """
    Validate the mac_shadow module
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current settings\n        '
        self.run_function('user.add', [TEST_USER])

    def tearDown(self):
        if False:
            return 10
        '\n        Reset to original settings\n        '
        self.run_function('user.delete', [TEST_USER])

    @pytest.mark.slow_test
    @pytest.mark.skip_initial_gh_actions_failure
    def test_info(self):
        if False:
            while True:
                i = 10
        '\n        Test shadow.info\n        '
        ret = self.run_function('shadow.info', [TEST_USER])
        self.assertEqual(ret['name'], TEST_USER)
        ret = self.run_function('shadow.info', [NO_USER])
        self.assertEqual(ret['name'], '')

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_get_account_created(self):
        if False:
            return 10
        '\n        Test shadow.get_account_created\n        '
        text_date = self.run_function('shadow.get_account_created', [TEST_USER])
        self.assertNotEqual(text_date, 'Invalid Timestamp')
        obj_date = datetime.datetime.strptime(text_date, '%Y-%m-%d %H:%M:%S')
        self.assertIsInstance(obj_date, datetime.date)
        self.assertEqual(self.run_function('shadow.get_account_created', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    @pytest.mark.skip_initial_gh_actions_failure
    def test_get_last_change(self):
        if False:
            print('Hello World!')
        '\n        Test shadow.get_last_change\n        '
        text_date = self.run_function('shadow.get_last_change', [TEST_USER])
        self.assertNotEqual(text_date, 'Invalid Timestamp')
        obj_date = datetime.datetime.strptime(text_date, '%Y-%m-%d %H:%M:%S')
        self.assertIsInstance(obj_date, datetime.date)
        self.assertEqual(self.run_function('shadow.get_last_change', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    @pytest.mark.skip_initial_gh_actions_failure
    def test_get_login_failed_last(self):
        if False:
            while True:
                i = 10
        '\n        Test shadow.get_login_failed_last\n        '
        text_date = self.run_function('shadow.get_login_failed_last', [TEST_USER])
        self.assertNotEqual(text_date, 'Invalid Timestamp')
        obj_date = datetime.datetime.strptime(text_date, '%Y-%m-%d %H:%M:%S')
        self.assertIsInstance(obj_date, datetime.date)
        self.assertEqual(self.run_function('shadow.get_login_failed_last', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    @pytest.mark.skip_initial_gh_actions_failure
    def test_get_login_failed_count(self):
        if False:
            i = 10
            return i + 15
        '\n        Test shadow.get_login_failed_count\n        '
        self.assertEqual(self.run_function('shadow.get_login_failed_count', [TEST_USER]), '0')
        self.assertEqual(self.run_function('shadow.get_login_failed_count', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_get_set_maxdays(self):
        if False:
            return 10
        '\n        Test shadow.get_maxdays\n        Test shadow.set_maxdays\n        '
        self.assertTrue(self.run_function('shadow.set_maxdays', [TEST_USER, 20]))
        self.assertEqual(self.run_function('shadow.get_maxdays', [TEST_USER]), 20)
        self.assertEqual(self.run_function('shadow.set_maxdays', [NO_USER, 7]), 'ERROR: User not found: {}'.format(NO_USER))
        self.assertEqual(self.run_function('shadow.get_maxdays', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_get_set_change(self):
        if False:
            return 10
        '\n        Test shadow.get_change\n        Test shadow.set_change\n        '
        self.assertTrue(self.run_function('shadow.set_change', [TEST_USER, '02/11/2011']))
        self.assertEqual(self.run_function('shadow.get_change', [TEST_USER]), '02/11/2011')
        self.assertEqual(self.run_function('shadow.set_change', [NO_USER, '02/11/2012']), 'ERROR: User not found: {}'.format(NO_USER))
        self.assertEqual(self.run_function('shadow.get_change', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_get_set_expire(self):
        if False:
            print('Hello World!')
        '\n        Test shadow.get_expire\n        Test shadow.set_expire\n        '
        self.assertTrue(self.run_function('shadow.set_expire', [TEST_USER, '02/11/2011']))
        self.assertEqual(self.run_function('shadow.get_expire', [TEST_USER]), '02/11/2011')
        self.assertEqual(self.run_function('shadow.set_expire', [NO_USER, '02/11/2012']), 'ERROR: User not found: {}'.format(NO_USER))
        self.assertEqual(self.run_function('shadow.get_expire', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_del_password(self):
        if False:
            return 10
        '\n        Test shadow.del_password\n        '
        self.assertTrue(self.run_function('shadow.del_password', [TEST_USER]))
        self.assertEqual(self.run_function('shadow.info', [TEST_USER])['passwd'], '*')
        self.assertEqual(self.run_function('shadow.del_password', [NO_USER]), 'ERROR: User not found: {}'.format(NO_USER))

    @pytest.mark.destructive_test
    @pytest.mark.slow_test
    def test_set_password(self):
        if False:
            print('Hello World!')
        '\n        Test shadow.set_password\n        '
        self.assertTrue(self.run_function('shadow.set_password', [TEST_USER, 'Pa$$W0rd']))
        self.assertEqual(self.run_function('shadow.set_password', [NO_USER, 'P@SSw0rd']), 'ERROR: User not found: {}'.format(NO_USER))