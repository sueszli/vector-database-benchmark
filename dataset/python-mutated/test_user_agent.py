import datetime as dt
import re
from typing import Pattern
from freezegun import freeze_time
from faker import Faker
from faker.providers.user_agent import Provider as UaProvider

class TestUserAgentProvider:
    """Test user agent provider methods"""
    num_samples = 1000
    android_token_pattern: Pattern = re.compile('Android (?P<android_version>\\d+(?:\\.\\d){0,2})')
    ios_token_pattern: Pattern = re.compile('^(?P<apple_device>.*?); CPU \\1 OS ' + '(?P<ios_version>\\d+(?:_\\d){0,2}) like Mac OS X')
    mac_token_pattern: Pattern = re.compile('Macintosh; (?P<mac_processor>.*?) Mac OS X 10_([5-9]|1[0-2])_(\\d)')
    one_day = dt.timedelta(1.0)

    def test_android_platform_token(self, faker, num_samples):
        if False:
            return 10
        for _ in range(num_samples):
            match = self.android_token_pattern.fullmatch(faker.android_platform_token())
            assert match.group('android_version') in UaProvider.android_versions

    def test_ios_platform_token(self, faker, num_samples):
        if False:
            return 10
        for _ in range(num_samples):
            match = self.ios_token_pattern.fullmatch(faker.ios_platform_token())
            assert match.group('apple_device') in UaProvider.apple_devices
            assert match.group('ios_version').replace('_', '.') in UaProvider.ios_versions

    def test_mac_platform_token(self, faker, num_samples):
        if False:
            i = 10
            return i + 15
        for _ in range(num_samples):
            match = self.mac_token_pattern.fullmatch(faker.mac_platform_token())
            assert match.group('mac_processor') in UaProvider.mac_processors

    def test_firefox_deterministic_output(self, faker: Faker, num_samples: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check whether ``faker.firefox()`` is deterministic, given the same seed.'
        for _ in range(num_samples):
            seed = faker.random.random()
            faker.seed_instance(seed)
            with freeze_time(dt.datetime.now() + self.one_day):
                fake_firefox_ua_output_tomorrow = faker.firefox()
            faker.seed_instance(seed)
            with freeze_time(dt.datetime.max - self.one_day):
                fake_firefox_ua_output_much_later = faker.firefox()
            assert fake_firefox_ua_output_much_later == fake_firefox_ua_output_tomorrow