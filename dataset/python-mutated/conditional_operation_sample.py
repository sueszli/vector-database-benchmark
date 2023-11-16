"""
FILE: conditional_operation_sample.py

DESCRIPTION:
    This sample demos conditional set/get/delete operations for app configuration

USAGE: python conditional_operation_sample.py

    Set the environment variables with your own values before running the sample:
    1) APPCONFIGURATION_CONNECTION_STRING: Connection String used to access the Azure App Configuration.
"""
import os
from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from util import print_configuration_setting

def main():
    if False:
        while True:
            i = 10
    CONNECTION_STRING = os.environ['APPCONFIGURATION_CONNECTION_STRING']
    client = AzureAppConfigurationClient.from_connection_string(CONNECTION_STRING)
    config_setting = ConfigurationSetting(key='MyKey', value='my value', content_type='my content type', tags={'my tag': 'my tag value'})
    client.set_configuration_setting(config_setting)
    first_get = client.get_configuration_setting(key='MyKey')
    if first_get is None:
        return print('Error, unconditional set failed.')
    print_configuration_setting(first_get)
    second_get = client.get_configuration_setting(key='MyKey', etag=first_get.etag, match_condition=MatchConditions.IfModified)
    print_configuration_setting(second_get)
    first_get.value = 'new value'
    client.set_configuration_setting(configuration_setting=first_get, match_condition=MatchConditions.IfNotModified)
    try:
        client.set_configuration_setting(configuration_setting=first_get, match_condition=MatchConditions.IfNotModified)
    except ResourceModifiedError:
        pass
    client.delete_configuration_setting(key='MyKey')
if __name__ == '__main__':
    main()