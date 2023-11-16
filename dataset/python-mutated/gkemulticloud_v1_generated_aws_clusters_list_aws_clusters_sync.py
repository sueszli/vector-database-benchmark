from google.cloud import gke_multicloud_v1

def sample_list_aws_clusters():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.ListAwsClustersRequest(parent='parent_value')
    page_result = client.list_aws_clusters(request=request)
    for response in page_result:
        print(response)