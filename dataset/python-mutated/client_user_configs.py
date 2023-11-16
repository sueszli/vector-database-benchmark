import azure.cosmos.cosmos_client as cosmos_client
import config
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']

def change_connection_retry_policy_configs():
    if False:
        i = 10
        return i + 15
    cosmos_client.CosmosClient(url=HOST, credential=MASTER_KEY, retry_total=10, retry_connect=3, retry_read=3, retry_status=3, retry_on_status_codes=[], retry_backoff_factor=0.08, retry_backoff_max=120, retry_fixed_interval=None)
    print('Client initialized with custom retry options')
if __name__ == '__main__':
    change_connection_retry_policy_configs()