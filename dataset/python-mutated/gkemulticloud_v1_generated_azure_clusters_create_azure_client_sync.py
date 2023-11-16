from google.cloud import gke_multicloud_v1

def sample_create_azure_client():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AzureClustersClient()
    azure_client = gke_multicloud_v1.AzureClient()
    azure_client.tenant_id = 'tenant_id_value'
    azure_client.application_id = 'application_id_value'
    request = gke_multicloud_v1.CreateAzureClientRequest(parent='parent_value', azure_client=azure_client, azure_client_id='azure_client_id_value')
    operation = client.create_azure_client(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)