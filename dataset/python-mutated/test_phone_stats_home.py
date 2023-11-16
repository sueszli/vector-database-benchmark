import synapse
from synapse.app.phone_stats_home import start_phone_stats_home
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.util import Clock
from tests.server import ThreadedMemoryReactorClock
from tests.unittest import HomeserverTestCase
FIVE_MINUTES_IN_SECONDS = 300
ONE_DAY_IN_SECONDS = 86400

class PhoneHomeR30V2TestCase(HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def _advance_to(self, desired_time_secs: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        now = self.hs.get_clock().time()
        assert now < desired_time_secs
        self.reactor.advance(desired_time_secs - now)

    def make_homeserver(self, reactor: ThreadedMemoryReactorClock, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        hs = super().make_homeserver(reactor, clock)
        assert not hs.config.metrics.report_stats
        start_phone_stats_home(hs)
        return hs

    def test_r30v2_minimum_usage(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Tests the minimum amount of interaction necessary for the R30v2 metric\n        to consider a user 'retained'.\n        "
        user_id = self.register_user('u1', 'secret!')
        access_token = self.login('u1', 'secret!')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=access_token)
        self.helper.send(room_id, 'message', tok=access_token)
        first_post_at = self.hs.get_clock().time()
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        store = self.hs.get_datastores().main
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        self.reactor.advance(31 * ONE_DAY_IN_SECONDS)
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        self.helper.send(room_id, 'message2', tok=access_token)
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 1, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        self._advance_to(first_post_at + 60 * ONE_DAY_IN_SECONDS - 5)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 1, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        self._advance_to(first_post_at + 60 * ONE_DAY_IN_SECONDS + 5)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})

    def test_r30v2_user_must_be_retained_for_at_least_a_month(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests that a newly-registered user must be retained for a whole month\n        before appearing in the R30v2 statistic, even if they post every day\n        during that time!\n        '
        headers = (('User-Agent', 'Element/1.1 (Linux; U; Android 9; MatrixAndroidSDK_X 0.0.1)'),)
        user_id = self.register_user('u1', 'secret!')
        access_token = self.login('u1', 'secret!', custom_headers=headers)
        room_id = self.helper.create_room_as(room_creator=user_id, tok=access_token, custom_headers=headers)
        self.helper.send(room_id, 'message', tok=access_token, custom_headers=headers)
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        store = self.hs.get_datastores().main
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        for _ in range(30):
            self.reactor.advance(ONE_DAY_IN_SECONDS - FIVE_MINUTES_IN_SECONDS)
            self.helper.send(room_id, "I'm still here", tok=access_token, custom_headers=headers)
            self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
            r30_results = self.get_success(store.count_r30v2_users())
            self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        self.reactor.advance(ONE_DAY_IN_SECONDS)
        self.helper.send(room_id, 'Still here!', tok=access_token, custom_headers=headers)
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 1, 'android': 1, 'electron': 0, 'ios': 0, 'web': 0})

    def test_r30v2_returning_dormant_users_not_counted(self) -> None:
        if False:
            print('Hello World!')
        '\n        Tests that dormant users (users inactive for a long time) do not\n        contribute to R30v2 when they return for just a single day.\n        This is a key difference between R30 and R30v2.\n        '
        headers = (('User-Agent', 'Riot/1.4 (iPhone; iOS 13; Scale/4.00)'),)
        user_id = self.register_user('u1', 'secret!')
        access_token = self.login('u1', 'secret!', custom_headers=headers)
        room_id = self.helper.create_room_as(room_creator=user_id, tok=access_token, custom_headers=headers)
        self.helper.send(room_id, 'message', tok=access_token, custom_headers=headers)
        self.reactor.advance(60 * ONE_DAY_IN_SECONDS)
        self.helper.send(room_id, 'message', tok=access_token, custom_headers=headers)
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        store = self.hs.get_datastores().main
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})
        self.reactor.advance(32 * ONE_DAY_IN_SECONDS)
        self.helper.send(room_id, 'message', tok=access_token, custom_headers=headers)
        self.reactor.advance(FIVE_MINUTES_IN_SECONDS)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 1, 'ios': 1, 'android': 0, 'electron': 0, 'web': 0})
        self.reactor.advance(27.5 * ONE_DAY_IN_SECONDS)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 1, 'ios': 1, 'android': 0, 'electron': 0, 'web': 0})
        self.reactor.advance(ONE_DAY_IN_SECONDS)
        r30_results = self.get_success(store.count_r30v2_users())
        self.assertEqual(r30_results, {'all': 0, 'android': 0, 'electron': 0, 'ios': 0, 'web': 0})