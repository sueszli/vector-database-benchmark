from google.cloud import gke_multicloud_v1

def sample_generate_aws_access_token():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AwsClustersClient()
    request = gke_multicloud_v1.GenerateAwsAccessTokenRequest(aws_cluster='aws_cluster_value')
    response = client.generate_aws_access_token(request=request)
    print(response)