from azure.appconfiguration.provider import load, SettingSelector
from devtools_testutils import AzureRecordedTestCase, recorded_by_proxy
from preparers import app_config_decorator
from testcase import AppConfigTestCase

class TestAppConfigurationProvider(AppConfigTestCase):

    @recorded_by_proxy
    @app_config_decorator
    def test_backoff(self, appconfiguration_connection_string, appconfiguration_keyvault_secret_url):
        if False:
            while True:
                i = 10
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        min_backoff = 30000
        assert min_backoff == client._refresh_timer._calculate_backoff()
        attempts = 2
        client._refresh_timer.attempts = attempts
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff >= min_backoff and backoff <= min_backoff * (1 << attempts)
        attempts = 3
        client._refresh_timer.attempts = attempts
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff >= min_backoff and backoff <= min_backoff * (1 << attempts)

    @recorded_by_proxy
    @app_config_decorator
    def test_backoff_max_attempts(self, appconfiguration_connection_string, appconfiguration_keyvault_secret_url):
        if False:
            return 10
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        min_backoff = 3000
        attempts = 30
        client._refresh_timer.attempts = attempts
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff >= min_backoff and backoff <= min_backoff * (1 << attempts)
        attempts = 31
        client._refresh_timer.attempts = attempts
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff >= min_backoff and backoff <= min_backoff * (1 << 30)

    @recorded_by_proxy
    @app_config_decorator
    def test_backoff_bounds(self, appconfiguration_connection_string, appconfiguration_keyvault_secret_url):
        if False:
            i = 10
            return i + 15
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url, refresh_interval=1)
        assert client._refresh_timer._min_backoff == 1
        assert client._refresh_timer._max_backoff == 1
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url, refresh_interval=45)
        assert client._refresh_timer._min_backoff == 30
        assert client._refresh_timer._max_backoff == 45
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url, refresh_interval=700)
        assert client._refresh_timer._min_backoff == 30
        assert client._refresh_timer._max_backoff == 600

    @recorded_by_proxy
    @app_config_decorator
    def test_backoff_invalid_attempts(self, appconfiguration_connection_string, appconfiguration_keyvault_secret_url):
        if False:
            return 10
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        min_backoff = 30000
        attempts = 0
        client._refresh_timer.attempts = attempts
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff == min_backoff
        attempts = -1
        client._refresh_timer.attempts = attempts
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff == min_backoff

    @recorded_by_proxy
    @app_config_decorator
    def test_backoff_missmatch_settings(self, appconfiguration_connection_string, appconfiguration_keyvault_secret_url):
        if False:
            while True:
                i = 10
        min_backoff = 30000
        client = self.create_client(appconfiguration_connection_string, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        client._refresh_timer.attempts = 0
        backoff = client._refresh_timer._calculate_backoff()
        assert backoff == min_backoff