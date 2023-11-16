import unittest
from mock import patch
from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler

class AdaptiveThrottlerTest(unittest.TestCase):
    START_TIME = 1500000000000
    SAMPLE_PERIOD = 60000
    BUCKET = 1000
    OVERLOAD_RATIO = 2

    def setUp(self):
        if False:
            while True:
                i = 10
        self._throttler = AdaptiveThrottler(AdaptiveThrottlerTest.SAMPLE_PERIOD, AdaptiveThrottlerTest.BUCKET, AdaptiveThrottlerTest.OVERLOAD_RATIO)

    def test_no_initial_throttling(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, self._throttler._throttling_probability(AdaptiveThrottlerTest.START_TIME))

    def test_no_throttling_if_no_errors(self):
        if False:
            i = 10
            return i + 15
        for t in range(AdaptiveThrottlerTest.START_TIME, AdaptiveThrottlerTest.START_TIME + 20):
            self.assertFalse(self._throttler.throttle_request(t))
            self._throttler.successful_request(t)
        self.assertEqual(0, self._throttler._throttling_probability(AdaptiveThrottlerTest.START_TIME + 20))

    def test_no_throttling_after_errors_expire(self):
        if False:
            for i in range(10):
                print('nop')
        for t in range(AdaptiveThrottlerTest.START_TIME, AdaptiveThrottlerTest.START_TIME + AdaptiveThrottlerTest.SAMPLE_PERIOD, 100):
            self._throttler.throttle_request(t)
        self.assertLess(0, self._throttler._throttling_probability(AdaptiveThrottlerTest.START_TIME + AdaptiveThrottlerTest.SAMPLE_PERIOD))
        for t in range(AdaptiveThrottlerTest.START_TIME + AdaptiveThrottlerTest.SAMPLE_PERIOD, AdaptiveThrottlerTest.START_TIME + AdaptiveThrottlerTest.SAMPLE_PERIOD * 2, 100):
            self._throttler.throttle_request(t)
            self._throttler.successful_request(t)
        self.assertEqual(0, self._throttler._throttling_probability(AdaptiveThrottlerTest.START_TIME + AdaptiveThrottlerTest.SAMPLE_PERIOD * 2))

    @patch('random.Random')
    def test_throttling_after_errors(self, mock_random):
        if False:
            for i in range(10):
                print('nop')
        mock_random().uniform.side_effect = [x / 10.0 for x in range(0, 10)] * 2
        self._throttler = AdaptiveThrottler(AdaptiveThrottlerTest.SAMPLE_PERIOD, AdaptiveThrottlerTest.BUCKET, AdaptiveThrottlerTest.OVERLOAD_RATIO)
        for t in range(AdaptiveThrottlerTest.START_TIME, AdaptiveThrottlerTest.START_TIME + 20):
            throttled = self._throttler.throttle_request(t)
            if t % 3 == 1:
                self._throttler.successful_request(t)
            if t > AdaptiveThrottlerTest.START_TIME + 10:
                self.assertAlmostEqual(0.33, self._throttler._throttling_probability(t), delta=0.1)
                self.assertEqual(t < AdaptiveThrottlerTest.START_TIME + 14, throttled)
if __name__ == '__main__':
    unittest.main()