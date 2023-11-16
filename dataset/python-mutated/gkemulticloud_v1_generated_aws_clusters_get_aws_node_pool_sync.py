from google.cloud import gke_multicloud_v1

def sample_get_aws_node_pool():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.GetAwsNodePoolRequest(name='name_value')
    response = client.get_aws_node_pool(request=request)
    print(response)