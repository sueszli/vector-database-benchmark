from google.cloud import redis_cluster_v1

def sample_get_cluster():
    if False:
        while True:
            i = 10
    client = redis_cluster_v1.CloudRedisClusterClient()
    request = redis_cluster_v1.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)