from r2.lib.utils.reddit_agent_parser import AlienBlueDetector, BaconReaderDetector, detect, McRedditDetector, NarwhalForRedditDetector, ReaditDetector, RedditAndroidDetector, RedditIsFunDetector, RedditIOSDetector, RedditSyncDetector, RelayForRedditDetector
from r2.tests import RedditTestCase

class AgentDetectorTest(RedditTestCase):

    def test_reddit_is_fun_detector(self):
        if False:
            print('Hello World!')
        user_agent = 'reddit is fun (Android) 4.1.15'
        agent_parsed = {}
        result = RedditIsFunDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], 'reddit is fun')
        self.assertEqual(agent_parsed['browser']['version'], '4.1.15')
        self.assertEqual(agent_parsed['platform']['name'], 'Android')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_reddit_android_detector(self):
        if False:
            return 10
        user_agent = 'RedditAndroid 1.1.5'
        agent_parsed = {}
        result = RedditAndroidDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], RedditAndroidDetector.name)
        self.assertEqual(agent_parsed['browser']['version'], '1.1.5')
        self.assertTrue(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_reddit_ios_detector(self):
        if False:
            i = 10
            return i + 15
        user_agent = 'Reddit/Version 1.1/Build 1106/iOS Version 9.3.2 (Build 13F69)'
        agent_parsed = {}
        result = RedditIOSDetector().detect(user_agent, agent_parsed)
        self.assertEqual(agent_parsed['browser']['name'], RedditIOSDetector.name)
        self.assertEqual(agent_parsed['browser']['version'], '1.1')
        self.assertEqual(agent_parsed['platform']['name'], 'iOS')
        self.assertEqual(agent_parsed['platform']['version'], '9.3.2')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_alian_blue_detector(self):
        if False:
            return 10
        user_agent = 'AlienBlue/2.9.10.0.2 CFNetwork/758.4.3 Darwin/15.5.0'
        agent_parsed = {}
        result = AlienBlueDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], AlienBlueDetector.name)
        self.assertEqual(agent_parsed['browser']['version'], '2.9.10.0.2')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['app_name'])

    def test_relay_for_reddit_detector(self):
        if False:
            i = 10
            return i + 15
        user_agent = 'Relay by /u/DBrady v7.9.32'
        agent_parsed = {}
        result = RelayForRedditDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], RelayForRedditDetector.name)
        self.assertEqual(agent_parsed['browser']['version'], '7.9.32')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_reddit_sync_detector(self):
        if False:
            return 10
        user_agent = 'android:com.laurencedawson.reddit_sync:v11.4 (by /u/ljdawson)'
        agent_parsed = {}
        result = RedditSyncDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], RedditSyncDetector.name)
        self.assertEqual(agent_parsed['browser']['version'], '11.4')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_narwhal_detector(self):
        if False:
            for i in range(10):
                print('nop')
        user_agent = 'narwhal-iOS/2306 by det0ur'
        agent_parsed = {}
        result = NarwhalForRedditDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], NarwhalForRedditDetector.name)
        self.assertEqual(agent_parsed['platform']['name'], 'iOS')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_mcreddit_detector(self):
        if False:
            print('Hello World!')
        user_agent = 'McReddit - Reddit Client for iOS'
        agent_parsed = {}
        result = McRedditDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], McRedditDetector.name)
        self.assertEqual(agent_parsed['platform']['name'], 'iOS')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

    def test_readit_detector(self):
        if False:
            return 10
        user_agent = '(Readit for WP /u/MessageAcrossStudios) (Readit for WP /u/MessageAcrossStudios)'
        agent_parsed = {}
        result = ReaditDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], ReaditDetector.name)
        self.assertIsNone(agent_parsed.get('app_name'))

    def test_bacon_reader_detector(self):
        if False:
            print('Hello World!')
        user_agent = 'BaconReader/3.0 (iPhone; iOS 9.3.2; Scale/2.00)'
        agent_parsed = {}
        result = BaconReaderDetector().detect(user_agent, agent_parsed)
        self.assertTrue(result)
        self.assertEqual(agent_parsed['browser']['name'], BaconReaderDetector.name)
        self.assertEqual(agent_parsed['browser']['version'], '3.0')
        self.assertEqual(agent_parsed['platform']['name'], 'iOS')
        self.assertEqual(agent_parsed['platform']['version'], '9.3.2')
        self.assertEqual(agent_parsed['app_name'], agent_parsed['browser']['name'])

class HAPIntegrationTests(RedditTestCase):
    """Tests to ensure that parsers don't confilct with existing onex."""

    def test_reddit_is_fun_integration(self):
        if False:
            print('Hello World!')
        user_agent = 'reddit is fun (Android) 4.1.15'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], 'reddit is fun')
        self.assertEqual(outs['dist']['name'], 'Android')

    def test_reddit_android_integration(self):
        if False:
            while True:
                i = 10
        user_agent = 'RedditAndroid 1.1.5'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], 'Reddit: The Official App')
        self.assertEqual(outs['dist']['name'], 'Android')

    def test_reddit_ios_integration(self):
        if False:
            i = 10
            return i + 15
        user_agent = 'Reddit/Version 1.1/Build 1106/iOS Version 9.3.2 (Build 13F69)'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], RedditIOSDetector.name)

    def test_alien_blue_detector(self):
        if False:
            i = 10
            return i + 15
        user_agent = 'AlienBlue/2.9.10.0.2 CFNetwork/758.4.3 Darwin/15.5.0'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], AlienBlueDetector.name)

    def test_relay_for_reddit_detector(self):
        if False:
            for i in range(10):
                print('nop')
        user_agent = '  Relay by /u/DBrady v7.9.32'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], RelayForRedditDetector.name)

    def test_reddit_sync_detector(self):
        if False:
            return 10
        user_agent = 'android:com.laurencedawson.reddit_sync:v11.4 (by /u/ljdawson)'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], RedditSyncDetector.name)

    def test_narwhal_detector(self):
        if False:
            i = 10
            return i + 15
        user_agent = 'narwhal-iOS/2306 by det0ur'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], NarwhalForRedditDetector.name)

    def test_mcreddit_detector(self):
        if False:
            print('Hello World!')
        user_agent = 'McReddit - Reddit Client for iOS'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], McRedditDetector.name)

    def test_readit_detector(self):
        if False:
            i = 10
            return i + 15
        user_agent = '(Readit for WP /u/MessageAcrossStudios) (Readit for WP /u/MessageAcrossStudios)'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], ReaditDetector.name)

    def test_bacon_reader_detector(self):
        if False:
            print('Hello World!')
        user_agent = 'BaconReader/3.0 (iPhone; iOS 9.3.2; Scale/2.00)'
        outs = detect(user_agent)
        self.assertEqual(outs['browser']['name'], BaconReaderDetector.name)