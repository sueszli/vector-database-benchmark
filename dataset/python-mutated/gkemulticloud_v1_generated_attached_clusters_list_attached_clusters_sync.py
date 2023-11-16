from google.cloud import gke_multicloud_v1

def sample_list_attached_clusters():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AttachedClustersClient()
    request = gke_multicloud_v1.ListAttachedClustersRequest(parent='parent_value')
    page_result = client.list_attached_clusters(request=request)
    for response in page_result:
        print(response)