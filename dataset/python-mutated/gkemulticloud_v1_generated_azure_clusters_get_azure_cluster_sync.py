from google.cloud import gke_multicloud_v1

def sample_get_azure_cluster():
    if False:
        return 10
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.GetAzureClusterRequest(name='name_value')
    response = client.get_azure_cluster(request=request)
    print(response)