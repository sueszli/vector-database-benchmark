from google.cloud import gke_multicloud_v1

def sample_delete_azure_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.DeleteAzureClusterRequest(name='name_value')
    operation = client.delete_azure_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)