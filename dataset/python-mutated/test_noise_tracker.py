from unittest import TestCase
from mycroft.client.speech.mic import NoiseTracker
LOUD_TIME_LIMIT = 2.0
SILENCE_TIME_LIMIT = 5.0
SECS_PER_BUFFER = 0.5
MIN_NOISE = 0
MAX_NOISE = 25

class TestNoiseTracker(TestCase):

    def test_no_loud_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that no loud data generates complete after silence timeout.'
        noise_tracker = NoiseTracker(MIN_NOISE, MAX_NOISE, SECS_PER_BUFFER, LOUD_TIME_LIMIT, SILENCE_TIME_LIMIT)
        num_updates_timeout = int(SILENCE_TIME_LIMIT / SECS_PER_BUFFER)
        num_low_updates = int(LOUD_TIME_LIMIT / SECS_PER_BUFFER)
        for _ in range(num_low_updates):
            noise_tracker.update(False)
            self.assertFalse(noise_tracker.recording_complete())
        remaining_until_low_timeout = num_updates_timeout - num_low_updates
        for _ in range(remaining_until_low_timeout):
            noise_tracker.update(False)
            self.assertFalse(noise_tracker.recording_complete())
        noise_tracker.update(False)
        self.assertTrue(noise_tracker.recording_complete())

    def test_silence_reset(self):
        if False:
            return 10
        'Check that no loud data generates complete after silence timeout.'
        noise_tracker = NoiseTracker(MIN_NOISE, MAX_NOISE, SECS_PER_BUFFER, LOUD_TIME_LIMIT, SILENCE_TIME_LIMIT)
        num_updates_timeout = int(SILENCE_TIME_LIMIT / SECS_PER_BUFFER)
        num_low_updates = int(LOUD_TIME_LIMIT / SECS_PER_BUFFER)
        for _ in range(num_low_updates):
            noise_tracker.update(False)
        noise_tracker.update(True)
        remaining_until_low_timeout = num_updates_timeout - num_low_updates
        for _ in range(remaining_until_low_timeout + 1):
            noise_tracker.update(False)
            self.assertFalse(noise_tracker.recording_complete())
        for _ in range(num_low_updates + 1):
            noise_tracker.update(False)
        self.assertTrue(noise_tracker.recording_complete())

    def test_all_loud_data(self):
        if False:
            for i in range(10):
                print('nop')
        "Check that only loud samples doesn't generate a complete recording.\n        "
        noise_tracker = NoiseTracker(MIN_NOISE, MAX_NOISE, SECS_PER_BUFFER, LOUD_TIME_LIMIT, SILENCE_TIME_LIMIT)
        num_high_updates = int(LOUD_TIME_LIMIT / SECS_PER_BUFFER) + 1
        for _ in range(num_high_updates):
            noise_tracker.update(True)
            self.assertFalse(noise_tracker.recording_complete())

    def test_all_loud_followed_by_silence(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that a long enough high sentence is completed after silence.\n        '
        noise_tracker = NoiseTracker(MIN_NOISE, MAX_NOISE, SECS_PER_BUFFER, LOUD_TIME_LIMIT, SILENCE_TIME_LIMIT)
        num_high_updates = int(LOUD_TIME_LIMIT / SECS_PER_BUFFER) + 1
        for _ in range(num_high_updates):
            noise_tracker.update(True)
            self.assertFalse(noise_tracker.recording_complete())
        while not noise_tracker._quiet_enough():
            noise_tracker.update(False)
        self.assertTrue(noise_tracker.recording_complete())