from google.cloud import dlp_v2

def sample_create_discovery_config():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    discovery_config = dlp_v2.DiscoveryConfig()
    discovery_config.status = 'PAUSED'
    request = dlp_v2.CreateDiscoveryConfigRequest(parent='parent_value', discovery_config=discovery_config)
    response = client.create_discovery_config(request=request)
    print(response)