from azure.appconfiguration.provider import SettingSelector, AzureAppConfigurationKeyVaultOptions
from devtools_testutils import recorded_by_proxy
from preparers import app_config_decorator_aad
from testcase import AppConfigTestCase

class TestAppConfigurationProvider(AppConfigTestCase):

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_creation_aad(self, appconfiguration_endpoint_string, appconfiguration_keyvault_secret_url):
        if False:
            i = 10
            return i + 15
        client = self.create_aad_client(appconfiguration_endpoint_string, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        assert client['message'] == 'hi'
        assert client['my_json']['key'] == 'value'
        assert 'FeatureManagementFeatureFlags' in client
        assert 'Alpha' in client['FeatureManagementFeatureFlags']

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_trim_prefixes(self, appconfiguration_endpoint_string, appconfiguration_keyvault_secret_url):
        if False:
            i = 10
            return i + 15
        trimmed = {'test.'}
        client = self.create_aad_client(appconfiguration_endpoint_string, trim_prefixes=trimmed, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        assert client['message'] == 'hi'
        assert client['my_json']['key'] == 'value'
        assert client['trimmed'] == 'key'
        assert 'test.trimmed' not in client
        assert 'FeatureManagementFeatureFlags' in client
        assert 'Alpha' in client['FeatureManagementFeatureFlags']

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_selectors(self, appconfiguration_endpoint_string, appconfiguration_keyvault_secret_url):
        if False:
            print('Hello World!')
        selects = {SettingSelector(key_filter='message*', label_filter='dev')}
        client = self.create_aad_client(appconfiguration_endpoint_string, selects=selects, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        assert client['message'] == 'test'
        assert 'test.trimmed' not in client
        assert 'FeatureManagementFeatureFlags' not in client

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_key_vault_reference(self, appconfiguration_endpoint_string, appconfiguration_keyvault_secret_url):
        if False:
            for i in range(10):
                print('nop')
        selects = {SettingSelector(key_filter='*', label_filter='prod')}
        client = self.create_aad_client(appconfiguration_endpoint_string, selects=selects, keyvault_secret_url=appconfiguration_keyvault_secret_url)
        assert client['secret'] == 'Very secret value'

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_secret_resolver(self, appconfiguration_endpoint_string):
        if False:
            for i in range(10):
                print('nop')
        selects = {SettingSelector(key_filter='*', label_filter='prod')}
        client = self.create_aad_client(appconfiguration_endpoint_string, selects=selects, secret_resolver=secret_resolver)
        assert client['secret'] == 'Reslover Value'

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_key_vault_reference_options(self, appconfiguration_endpoint_string, appconfiguration_keyvault_secret_url):
        if False:
            i = 10
            return i + 15
        selects = {SettingSelector(key_filter='*', label_filter='prod')}
        key_vault_options = AzureAppConfigurationKeyVaultOptions()
        client = self.create_aad_client(appconfiguration_endpoint_string, selects=selects, keyvault_secret_url=appconfiguration_keyvault_secret_url, key_vault_options=key_vault_options)
        assert client['secret'] == 'Very secret value'

    @recorded_by_proxy
    @app_config_decorator_aad
    def test_provider_secret_resolver_options(self, appconfiguration_endpoint_string):
        if False:
            i = 10
            return i + 15
        selects = {SettingSelector(key_filter='*', label_filter='prod')}
        key_vault_options = AzureAppConfigurationKeyVaultOptions(secret_resolver=secret_resolver)
        client = self.create_aad_client(appconfiguration_endpoint_string, selects=selects, key_vault_options=key_vault_options)
        assert client['secret'] == 'Reslover Value'

def secret_resolver(secret_id):
    if False:
        for i in range(10):
            print('nop')
    return 'Reslover Value'