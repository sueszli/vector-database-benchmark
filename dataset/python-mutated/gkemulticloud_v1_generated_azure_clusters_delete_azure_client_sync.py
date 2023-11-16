from google.cloud import gke_multicloud_v1

def sample_delete_azure_client():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.DeleteAzureClientRequest(name='name_value')
    operation = client.delete_azure_client(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)