from google.cloud import gke_multicloud_v1

def sample_get_azure_client():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.GetAzureClientRequest(name='name_value')
    response = client.get_azure_client(request=request)
    print(response)