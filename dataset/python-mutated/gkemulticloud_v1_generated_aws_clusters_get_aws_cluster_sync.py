from google.cloud import gke_multicloud_v1

def sample_get_aws_cluster():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.GetAwsClusterRequest(name='name_value')
    response = client.get_aws_cluster(request=request)
    print(response)