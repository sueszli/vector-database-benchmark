"""
FILE: hello_world_advanced_sample.py

DESCRIPTION:
    This sample demos more advanced scenarios including add/set with label/list operations for app configuration

USAGE: python hello_world_advanced_sample.py

    Set the environment variables with your own values before running the sample:
    1) APPCONFIGURATION_CONNECTION_STRING: Connection String used to access the Azure App Configuration.
"""
import os
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from util import print_configuration_setting

def main():
    if False:
        i = 10
        return i + 15
    CONNECTION_STRING = os.environ['APPCONFIGURATION_CONNECTION_STRING']
    client = AzureAppConfigurationClient.from_connection_string(CONNECTION_STRING)
    print('Add new configuration setting')
    config_setting = ConfigurationSetting(key='MyKey', label='MyLabel', value='my value', content_type='my content type', tags={'my tag': 'my tag value'})
    added_config_setting = client.add_configuration_setting(config_setting)
    print('New configuration setting:')
    print_configuration_setting(added_config_setting)
    print('')
    print('Set configuration setting')
    added_config_setting.value = 'new value'
    added_config_setting.content_type = 'new content type'
    updated_config_setting = client.set_configuration_setting(added_config_setting)
    print_configuration_setting(updated_config_setting)
    print('')
    print('Get configuration setting')
    fetched_config_setting = client.get_configuration_setting(key='MyKey', label='MyLabel')
    print('Fetched configuration setting:')
    print_configuration_setting(fetched_config_setting)
    print('')
    print('List configuration settings')
    config_settings = client.list_configuration_settings(label_filter='MyLabel')
    for item in config_settings:
        print_configuration_setting(item)
    print('Delete configuration setting')
    client.delete_configuration_setting(key='MyKey', label='MyLabel')
if __name__ == '__main__':
    main()