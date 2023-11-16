from google.cloud import gke_multicloud_v1

def sample_generate_azure_access_token():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.GenerateAzureAccessTokenRequest(azure_cluster='azure_cluster_value')
    response = client.generate_azure_access_token(request=request)
    print(response)