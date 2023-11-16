from google.cloud import redis_cluster_v1beta1

def sample_get_cluster():
    if False:
        i = 10
        return i + 15
    client = redis_cluster_v1beta1.CloudRedisClusterClient()
    request = redis_cluster_v1beta1.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)