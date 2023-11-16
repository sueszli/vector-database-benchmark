from azure.appconfiguration.provider import load, WatchKey
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from sample_utilities import get_client_modifications
import os
import time
kwargs = get_client_modifications()
connection_string = os.environ.get('APPCONFIGURATION_CONNECTION_STRING')
client = AzureAppConfigurationClient.from_connection_string(connection_string)
configuration_setting = ConfigurationSetting(key='message', value='Hello World!')
client.set_configuration_setting(configuration_setting=configuration_setting)

def my_callback_on_fail(error):
    if False:
        for i in range(10):
            print('nop')
    print('Refresh failed!')
config = load(connection_string=connection_string, refresh_on=[WatchKey('message')], refresh_interval=1, on_refresh_error=my_callback_on_fail, **kwargs)
print(config['message'])
print(config['my_json']['key'])
configuration_setting.value = 'Hello World Updated!'
client.set_configuration_setting(configuration_setting=configuration_setting)
time.sleep(2)
config.refresh()
print(config['message'])
print(config['my_json']['key'])
time.sleep(2)
config.refresh()
print(config['message'])
print(config['my_json']['key'])