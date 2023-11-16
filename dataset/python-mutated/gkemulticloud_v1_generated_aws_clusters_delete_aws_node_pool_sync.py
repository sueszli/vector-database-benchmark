from google.cloud import gke_multicloud_v1

def sample_delete_aws_node_pool():
    if False:
        for i in range(10):
            print('nop')
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.DeleteAwsNodePoolRequest(name='name_value')
    operation = client.delete_aws_node_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)