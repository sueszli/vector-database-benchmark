from google.cloud import gke_multicloud_v1

def sample_list_azure_clusters():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AzureClustersClient()
    request = gke_multicloud_v1.ListAzureClustersRequest(parent='parent_value')
    page_result = client.list_azure_clusters(request=request)
    for response in page_result:
        print(response)