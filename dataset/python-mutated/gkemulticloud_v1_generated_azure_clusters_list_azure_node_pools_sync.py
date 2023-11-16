from google.cloud import gke_multicloud_v1

def sample_list_azure_node_pools():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.ListAzureNodePoolsRequest(parent='parent_value')
    page_result = client.list_azure_node_pools(request=request)
    for response in page_result:
        print(response)