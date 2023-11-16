from google.cloud import gke_multicloud_v1

def sample_list_aws_node_pools():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.ListAwsNodePoolsRequest(parent='parent_value')
    page_result = client.list_aws_node_pools(request=request)
    for response in page_result:
        print(response)