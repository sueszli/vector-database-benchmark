from google.cloud import redis_cluster_v1

def sample_list_clusters():
    if False:
        for i in range(10):
            print('nop')
    client = redis_cluster_v1.CloudRedisClusterClient()
    request = redis_cluster_v1.ListClustersRequest(parent='parent_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)