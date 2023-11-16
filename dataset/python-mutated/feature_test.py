import collections
import random
import string
import unittest
import mock
from r2.config.feature.state import FeatureState
from r2.config.feature.world import World
from r2.tests import RedditTestCase

class MockAccount(object):

    def __init__(self, name, _fullname):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self._fullname = _fullname
        (_, _, _id) = _fullname.partition('_')
        self._id = int(_id, 36)
gary = MockAccount(name='gary', _fullname='t2_beef')
all_uppercase = MockAccount(name='ALL_UPPERCASE', _fullname='t2_f00d')

class MockWorld(World):

    def _make_state(self, config):
        if False:
            for i in range(10):
                print('nop')

        class MockState(FeatureState):

            def _parse_config(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                return config
        return MockState('test_state', self)

class TestFeatureBase(RedditTestCase):
    _world = None
    longMessage = True

class TestFeatureBase(RedditTestCase):
    longMessage = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.world = MockWorld()
        self.world.current_user = mock.Mock(return_value='')
        self.world.current_subreddit = mock.Mock(return_value='')
        self.world.current_loid = mock.Mock(return_value='')

class TestFeatureBase(RedditTestCase):
    longMessage = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestFeatureBase, self).setUp()
        self.world = MockWorld()
        self.world.current_user = mock.Mock(return_value='')
        self.world.current_subreddit = mock.Mock(return_value='')
        self.world.current_loid = mock.Mock(return_value='')

    @classmethod
    def generate_loid(cls):
        if False:
            while True:
                i = 10
        return ''.join(random.sample(string.letters + string.digits, 16))

class TestFeature(TestFeatureBase):

    def _assert_fuzzy_percent_true(self, results, percent):
        if False:
            i = 10
            return i + 15
        stats = collections.Counter(results)
        total = sum(stats.values())
        diff = abs(float(stats[True]) / total - percent / 100.0)
        self.assertTrue(diff < 0.1)

    def test_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = {'enabled': 'on'}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled())
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_disabled(self):
        if False:
            return 10
        cfg = {'enabled': 'off'}
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled())
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_admin_enabled(self):
        if False:
            while True:
                i = 10
        cfg = {'admin': True}
        self.world.is_admin = mock.Mock(return_value=True)
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_admin_disabled(self):
        if False:
            return 10
        cfg = {'admin': True}
        self.world.is_admin = mock.Mock(return_value=False)
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_employee_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = {'employee': True}
        self.world.is_employee = mock.Mock(return_value=True)
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_employee_disabled(self):
        if False:
            while True:
                i = 10
        cfg = {'employee': True}
        self.world.is_employee = mock.Mock(return_value=False)
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_beta_enabled(self):
        if False:
            return 10
        cfg = {'beta': True}
        self.world.user_has_beta_enabled = mock.Mock(return_value=True)
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_beta_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = {'beta': True}
        self.world.user_has_beta_enabled = mock.Mock(return_value=False)
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_gold_enabled(self):
        if False:
            i = 10
            return i + 15
        cfg = {'gold': True}
        self.world.has_gold = mock.Mock(return_value=True)
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_gold_disabled(self):
        if False:
            return 10
        cfg = {'gold': True}
        self.world.has_gold = mock.Mock(return_value=False)
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_loggedin_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = {'loggedin': True}
        self.world.is_user_loggedin = mock.Mock(return_value=True)
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_loggedin_disabled(self):
        if False:
            i = 10
            return i + 15
        cfg = {'loggedin': False}
        self.world.is_user_loggedin = mock.Mock(return_value=True)
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_loggedout_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = {'loggedout': True}
        self.world.is_user_loggedin = mock.Mock(return_value=False)
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_loggedout_disabled(self):
        if False:
            print('Hello World!')
        cfg = {'loggedout': False}
        self.world.is_user_loggedin = mock.Mock(return_value=False)
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(user=gary))

    def test_percent_loggedin(self):
        if False:
            for i in range(10):
                print('nop')
        num_users = 2000
        users = []
        for i in xrange(num_users):
            users.append(MockAccount(name=str(i), _fullname='t2_%s' % str(i)))

        def simulate_percent_loggedin(wanted_percent):
            if False:
                i = 10
                return i + 15
            cfg = {'percent_loggedin': wanted_percent}
            self.world.is_user_loggedin = mock.Mock(return_value=True)
            feature_state = self.world._make_state(cfg)
            return (feature_state.is_enabled(x) for x in users)
        self.assertFalse(any(simulate_percent_loggedin(0)))
        self.assertTrue(all(simulate_percent_loggedin(100)))
        self._assert_fuzzy_percent_true(simulate_percent_loggedin(25), 25)
        self._assert_fuzzy_percent_true(simulate_percent_loggedin(10), 10)
        self._assert_fuzzy_percent_true(simulate_percent_loggedin(50), 50)
        self._assert_fuzzy_percent_true(simulate_percent_loggedin(99), 99)

    def test_percent_loggedout(self):
        if False:
            while True:
                i = 10
        num_users = 2000

        def simulate_percent_loggedout(wanted_percent):
            if False:
                print('Hello World!')
            cfg = {'percent_loggedout': wanted_percent}
            for i in xrange(num_users):
                loid = self.generate_loid()
                self.world.current_loid = mock.Mock(return_value=loid)
                self.world.is_user_loggedin = mock.Mock(return_value=False)
                feature_state = self.world._make_state(cfg)
                yield feature_state.is_enabled()
        self.assertFalse(any(simulate_percent_loggedout(0)))
        self.assertTrue(all(simulate_percent_loggedout(100)))
        self._assert_fuzzy_percent_true(simulate_percent_loggedout(25), 25)
        self._assert_fuzzy_percent_true(simulate_percent_loggedout(10), 10)
        self._assert_fuzzy_percent_true(simulate_percent_loggedout(50), 50)
        self._assert_fuzzy_percent_true(simulate_percent_loggedout(99), 99)

    def test_url_enabled(self):
        if False:
            while True:
                i = 10
        cfg = {'url': 'test_state'}
        self.world.url_features = mock.Mock(return_value={'test_state'})
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled())
        self.assertTrue(feature_state.is_enabled(user=gary))
        cfg = {'url': 'test_state'}
        self.world.url_features = mock.Mock(return_value={'x', 'test_state'})
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled())
        self.assertTrue(feature_state.is_enabled(user=gary))
        cfg = {'url': {'test_state_a': 'a', 'test_state_b': 'b'}}
        self.world.url_features = mock.Mock(return_value={'x', 'test_state_b'})
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled())
        self.assertEqual(feature_state.variant(user=gary), 'b')

    def test_url_disabled(self):
        if False:
            return 10
        cfg = {'url': 'test_state'}
        self.world.url_features = mock.Mock(return_value={})
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled())
        self.assertFalse(feature_state.is_enabled(user=gary))
        cfg = {'url': 'test_state'}
        self.world.url_features = mock.Mock(return_value={'x'})
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled())
        self.assertFalse(feature_state.is_enabled(user=gary))
        cfg = {'url': {'test_state_a': 'a', 'test_state_b': 'b'}}
        self.world.url_features = mock.Mock(return_value={'x'})
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled())
        cfg = {'url': {'test_state_c1': 'control_1', 'test_state_c2': 'control_2'}}
        self.world.url_features = mock.Mock(return_value={'x', 'test_state_c2'})
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled())

    def test_user_in(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = {'users': ['Gary']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))
        cfg = {'users': ['ALL_UPPERCASE']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=all_uppercase))
        cfg = {'users': ['dave', 'gary']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(user=gary))

    def test_user_not_in(self):
        if False:
            return 10
        cfg = {'users': ['']}
        featurestate = self.world._make_state(cfg)
        self.assertFalse(featurestate.is_enabled(user=gary))
        cfg = {'users': ['dave', 'joe']}
        featurestate = self.world._make_state(cfg)
        self.assertFalse(featurestate.is_enabled(user=gary))

    def test_subreddit_in(self):
        if False:
            print('Hello World!')
        cfg = {'subreddits': ['WTF']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(subreddit='wtf'))
        cfg = {'subreddits': ['wtf']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(subreddit='WTF'))
        cfg = {'subreddits': ['aww', 'wtf']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(subreddit='wtf'))

    def test_subreddit_not_in(self):
        if False:
            while True:
                i = 10
        cfg = {'subreddits': []}
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(subreddit='wtf'))
        cfg = {'subreddits': ['aww', 'wtfoobar']}
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(subreddit='wtf'))

    def test_subdomain_in(self):
        if False:
            while True:
                i = 10
        cfg = {'subdomains': ['BETA']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(subdomain='beta'))
        cfg = {'subdomains': ['beta']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(subdomain='BETA'))
        cfg = {'subdomains': ['www', 'beta']}
        feature_state = self.world._make_state(cfg)
        self.assertTrue(feature_state.is_enabled(subdomain='beta'))

    def test_subdomain_not_in(self):
        if False:
            while True:
                i = 10
        cfg = {'subdomains': []}
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(subdomain='beta'))
        self.assertFalse(feature_state.is_enabled(subdomain=''))
        cfg = {'subdomains': ['www', 'betanauts']}
        feature_state = self.world._make_state(cfg)
        self.assertFalse(feature_state.is_enabled(subdomain='beta'))

    def test_multiple(self):
        if False:
            i = 10
            return i + 15
        cfg = {'enabled': 'off', 'admin': True}
        self.world.is_admin = mock.Mock(return_value=True)
        featurestate = self.world._make_state(cfg)
        self.assertFalse(featurestate.is_enabled(user=gary))
        cfg = {'enabled': 'on', 'admin': True}
        self.world.is_admin = mock.Mock(return_value=False)
        featurestate = self.world._make_state(cfg)
        self.assertTrue(featurestate.is_enabled(user=gary))
        self.assertTrue(featurestate.is_enabled())
        cfg = {'url': 'test_featurestate', 'admin': True}
        self.world.url_features = mock.Mock(return_value={})
        self.world.is_admin = mock.Mock(return_value=True)
        featurestate = self.world._make_state(cfg)
        self.assertTrue(featurestate.is_enabled(user=gary))