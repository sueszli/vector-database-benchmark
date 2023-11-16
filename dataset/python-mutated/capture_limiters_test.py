import unittest
import pandas as pd
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive.options.capture_limiters import CountLimiter
from apache_beam.runners.interactive.options.capture_limiters import ProcessingTimeLimiter
from apache_beam.utils.windowed_value import WindowedValue

class CaptureLimitersTest(unittest.TestCase):

    def test_count_limiter(self):
        if False:
            print('Hello World!')
        limiter = CountLimiter(5)
        for e in range(4):
            limiter.update(e)
        self.assertFalse(limiter.is_triggered())
        limiter.update(4)
        self.assertTrue(limiter.is_triggered())

    def test_count_limiter_with_dataframes(self):
        if False:
            for i in range(10):
                print('nop')
        limiter = CountLimiter(5)
        for _ in range(10):
            df = WindowedValue(pd.DataFrame(), 0, [])
            limiter.update(df)
        self.assertFalse(limiter.is_triggered())
        df = WindowedValue(pd.DataFrame({'col': list(range(10))}), 0, [])
        limiter.update(df)
        self.assertTrue(limiter.is_triggered())

    def test_processing_time_limiter(self):
        if False:
            print('Hello World!')
        limiter = ProcessingTimeLimiter(max_duration_secs=2)
        e = beam_runner_api_pb2.TestStreamPayload.Event()
        e.processing_time_event.advance_duration = int(1 * 1000000.0)
        limiter.update(e)
        self.assertFalse(limiter.is_triggered())
        e = beam_runner_api_pb2.TestStreamPayload.Event()
        e.processing_time_event.advance_duration = int(2 * 1000000.0)
        limiter.update(e)
        self.assertTrue(limiter.is_triggered())
if __name__ == '__main__':
    unittest.main()