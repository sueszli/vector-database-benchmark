from google.cloud import gke_multicloud_v1

def sample_delete_aws_cluster():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.DeleteAwsClusterRequest(name='name_value')
    operation = client.delete_aws_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)