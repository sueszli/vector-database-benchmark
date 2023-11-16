"""
FILE: hello_world_sample.py

DESCRIPTION:
    This sample demos set/get/delete operations for app configuration

USAGE: python hello_world_sample.py

    Set the environment variables with your own values before running the sample:
    1) APPCONFIGURATION_CONNECTION_STRING: Connection String used to access the Azure App Configuration.
"""
from azure.appconfiguration import ConfigurationSetting
from util import print_configuration_setting

def main():
    if False:
        return 10
    import os
    from azure.appconfiguration import AzureAppConfigurationClient
    CONNECTION_STRING = os.environ['APPCONFIGURATION_CONNECTION_STRING']
    client = AzureAppConfigurationClient.from_connection_string(CONNECTION_STRING)
    print('Set new configuration setting')
    config_setting = ConfigurationSetting(key='MyKey', value='my value', content_type='my content type', tags={'my tag': 'my tag value'})
    returned_config_setting = client.set_configuration_setting(config_setting)
    print('New configuration setting:')
    print_configuration_setting(returned_config_setting)
    print('')
    print('Get configuration setting')
    fetched_config_setting = client.get_configuration_setting(key='MyKey')
    print('Fetched configuration setting:')
    print_configuration_setting(fetched_config_setting)
    print('')
    print('Delete configuration setting')
    client.delete_configuration_setting(key='MyKey')
if __name__ == '__main__':
    main()