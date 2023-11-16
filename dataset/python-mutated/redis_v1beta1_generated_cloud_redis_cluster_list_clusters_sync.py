from google.cloud import redis_cluster_v1beta1

def sample_list_clusters():
    if False:
        i = 10
        return i + 15
    client = redis_cluster_v1beta1.CloudRedisClusterClient()
    request = redis_cluster_v1beta1.ListClustersRequest(parent='parent_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)