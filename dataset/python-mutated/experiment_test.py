import collections
import itertools
import math
from mock import MagicMock
from pylons import app_globals as g
from r2.config.feature.state import FeatureState
from .feature_test import TestFeatureBase, MockAccount

class TestExperiment(TestFeatureBase):
    _world = None
    longMessage = True

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestExperiment, self).setUp()
        self.world.is_user_loggedin = bool
        self.mock_eventcollector()
        self.patch_g(enable_loggedout_experiments=True)

    def get_loggedin_users(self, num_users):
        if False:
            for i in range(10):
                print('nop')
        users = []
        for i in xrange(num_users):
            users.append(MockAccount(name=str(i), _fullname='t2_%s' % str(i)))
        return users

    @staticmethod
    def get_loggedout_users(num_users):
        if False:
            return 10
        return [None for _ in xrange(num_users)]

    def test_calculate_bucket(self):
        if False:
            while True:
                i = 10
        "Test FeatureState's _calculate_bucket function."
        feature_state = self.world._make_state(config={})
        NUM_USERS = FeatureState.NUM_BUCKETS * 2000
        fullnames = []
        for i in xrange(NUM_USERS):
            fullnames.append('t2_%s' % str(i))
        counter = collections.Counter()
        for fullname in fullnames:
            bucket = feature_state._calculate_bucket(fullname)
            counter[bucket] += 1
            self.assertEqual(bucket, feature_state._calculate_bucket(fullname))
        for bucket in xrange(FeatureState.NUM_BUCKETS):
            expected = NUM_USERS / FeatureState.NUM_BUCKETS
            actual = counter[bucket]
            percent_equal = float(actual) / expected
            self.assertAlmostEqual(percent_equal, 1.0, delta=0.1, msg='bucket: %s' % bucket)

    def test_choose_variant(self):
        if False:
            while True:
                i = 10
        "Test FeatureState's _choose_variant function."
        no_variants = {}
        three_variants = {'remove_vote_counters': 5, 'control_1': 10, 'control_2': 5}
        three_variants_more = {'remove_vote_counters': 15.6, 'control_1': 10, 'control_2': 20}
        counters = collections.defaultdict(collections.Counter)
        for bucket in xrange(FeatureState.NUM_BUCKETS):
            variant = FeatureState._choose_variant(bucket, no_variants)
            if variant:
                counters['no_variants'][variant] += 1
            self.assertEqual(variant, FeatureState._choose_variant(bucket, no_variants))
            variant = FeatureState._choose_variant(bucket, three_variants)
            if variant:
                counters['three_variants'][variant] += 1
            self.assertEqual(variant, FeatureState._choose_variant(bucket, three_variants))
            previous_variant = variant
            variant = FeatureState._choose_variant(bucket, three_variants_more)
            if variant:
                counters['three_variants_more'][variant] += 1
            self.assertEqual(variant, FeatureState._choose_variant(bucket, three_variants_more))
            if previous_variant:
                self.assertEqual(variant, previous_variant)
        for (variant, percentage) in FeatureState.DEFAULT_CONTROL_GROUPS.items():
            count = counters['no_variants'][variant]
            scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
            self.assertEqual(scaled_percentage, percentage)
        for (variant, percentage) in three_variants.items():
            count = counters['three_variants'][variant]
            scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
            self.assertEqual(scaled_percentage, percentage)
        for (variant, percentage) in three_variants_more.items():
            count = counters['three_variants_more'][variant]
            scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
            self.assertEqual(scaled_percentage, percentage)
        fifty_fifty = {'control_1': 50, 'control_2': 50}
        almost_fifty_fifty = {'control_1': 49, 'control_2': 51}
        for bucket in xrange(FeatureState.NUM_BUCKETS):
            variant = FeatureState._choose_variant(bucket, fifty_fifty)
            counters['fifty_fifty'][variant] += 1
            variant = FeatureState._choose_variant(bucket, almost_fifty_fifty)
            counters['almost_fifty_fifty'][variant] += 1
        count = counters['fifty_fifty']['control_1']
        scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
        self.assertEqual(scaled_percentage, 50)
        count = counters['fifty_fifty']['control_2']
        scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
        self.assertEqual(scaled_percentage, 50)
        count = counters['almost_fifty_fifty']['control_1']
        scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
        self.assertEqual(scaled_percentage, 49)
        count = counters['almost_fifty_fifty']['control_2']
        scaled_percentage = float(count) / (FeatureState.NUM_BUCKETS / 100)
        self.assertEqual(scaled_percentage, 50)

    def do_experiment_simulation(self, users, loid_generator=None, **cfg):
        if False:
            while True:
                i = 10
        num_users = len(users)
        if loid_generator is None:
            loid_generator = iter(self.generate_loid, None)
        feature_state = self.world._make_state(cfg)
        counter = collections.Counter()
        for (user, loid) in zip(users, loid_generator):
            self.world.current_loid.return_value = loid
            variant = feature_state.variant(user)
            if feature_state.is_enabled(user):
                self.assertIsNotNone(variant, 'an enabled experiment should have a variant!')
                counter[variant] += 1
        error_bar_percent = 100.0 / math.sqrt(num_users)
        for (variant, percent) in cfg['experiment']['variants'].items():
            measured_percent = float(counter[variant]) / num_users * 100
            self.assertAlmostEqual(measured_percent, percent, delta=error_bar_percent)

    def assert_no_experiment(self, users, **cfg):
        if False:
            i = 10
            return i + 15
        feature_state = self.world._make_state(cfg)
        for user in users:
            self.assertFalse(feature_state.is_enabled(user))

    def test_loggedin_experiment(self, num_users=2000):
        if False:
            return 10
        'Test variant distn for logged in users.'
        self.do_experiment_simulation(self.get_loggedin_users(num_users), experiment={'loggedin': True, 'variants': {'larger': 5, 'smaller': 10}})

    def test_loggedin_experiment_explicit_enable(self, num_users=2000):
        if False:
            i = 10
            return i + 15
        'Test variant distn for logged in users with explicit enable.'
        self.do_experiment_simulation(self.get_loggedin_users(num_users), experiment={'loggedin': True, 'variants': {'larger': 5, 'smaller': 10}, 'enabled': True})

    def test_loggedin_experiment_explicit_disable(self, num_users=2000):
        if False:
            for i in range(10):
                print('nop')
        'Test explicit disable for logged in users actually disables.'
        self.assert_no_experiment(self.get_loggedin_users(num_users), experiment={'loggedin': True, 'variants': {'larger': 5, 'smaller': 10}, 'enabled': False})

    def test_loggedout_experiment(self, num_users=2000):
        if False:
            return 10
        'Test variant distn for logged out users.'
        self.do_experiment_simulation(self.get_loggedout_users(num_users), experiment={'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}})

    def test_loggedout_experiment_missing_loids(self, num_users=2000):
        if False:
            i = 10
            return i + 15
        'Ensure logged out experiments with no loids do not bucket.'
        self.assert_no_experiment(self.get_loggedout_users(num_users), loid_generator=itertools.repeat(None), experiment={'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}})

    def test_loggedout_experiment_explicit_enable(self, num_users=2000):
        if False:
            while True:
                i = 10
        'Test variant distn for logged out users with explicit enable.'
        self.do_experiment_simulation(self.get_loggedout_users(num_users), experiment={'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}, 'enabled': True})

    def test_loggedout_experiment_explicit_disable(self, num_users=2000):
        if False:
            for i in range(10):
                print('nop')
        'Test explicit disable for logged in users actually disables.'
        self.assert_no_experiment(self.get_loggedout_users(num_users), experiment={'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}, 'enabled': False})

    def test_loggedout_experiment_global_disable(self, num_users=2000):
        if False:
            while True:
                i = 10
        'Test we can disable loid-experiments via configuration.'
        g.enable_loggedout_experiments = False
        self.assert_no_experiment(self.get_loggedout_users(num_users), experiment={'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}, 'enabled': True})

    def test_mixed_experiment(self, num_users=2000):
        if False:
            while True:
                i = 10
        'Test a combination of loggedin/out users balances variants.'
        self.do_experiment_simulation(self.get_loggedin_users(num_users / 2) + self.get_loggedout_users(num_users / 2), experiment={'loggedin': True, 'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}})

    def test_mixed_experiment_disable(self, num_users=2000):
        if False:
            print('Hello World!')
        'Test a combination of loggedin/out users disables properly.'
        self.assert_no_experiment(self.get_loggedin_users(num_users / 2) + self.get_loggedout_users(num_users / 2), experiment={'loggedin': True, 'loggedout': True, 'variants': {'larger': 5, 'smaller': 10}, 'enabled': False})