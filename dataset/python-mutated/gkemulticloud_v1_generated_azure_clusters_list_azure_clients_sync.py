from google.cloud import gke_multicloud_v1

def sample_list_azure_clients():
    if False:
        for i in range(10):
            print('nop')
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.ListAzureClientsRequest(parent='parent_value')
    page_result = client.list_azure_clients(request=request)
    for response in page_result:
        print(response)