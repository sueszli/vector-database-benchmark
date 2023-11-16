from google.cloud import dlp_v2

def sample_delete_discovery_config():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeleteDiscoveryConfigRequest(name='name_value')
    client.delete_discovery_config(request=request)