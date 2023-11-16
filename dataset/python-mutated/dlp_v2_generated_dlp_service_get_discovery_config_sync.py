from google.cloud import dlp_v2

def sample_get_discovery_config():
    if False:
        while True:
            i = 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.GetDiscoveryConfigRequest(name='name_value')
    response = client.get_discovery_config(request=request)
    print(response)