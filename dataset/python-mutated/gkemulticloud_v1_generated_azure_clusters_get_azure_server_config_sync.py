from google.cloud import gke_multicloud_v1

def sample_get_azure_server_config():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.GetAzureServerConfigRequest(name='name_value')
    response = client.get_azure_server_config(request=request)
    print(response)