from google.cloud import dlp_v2

def sample_update_discovery_config():
    if False:
        i = 10
        return i + 15
    client = dlp_v2.DlpServiceClient()
    discovery_config = dlp_v2.DiscoveryConfig()
    discovery_config.status = 'PAUSED'
    request = dlp_v2.UpdateDiscoveryConfigRequest(name='name_value', discovery_config=discovery_config)
    response = client.update_discovery_config(request=request)
    print(response)