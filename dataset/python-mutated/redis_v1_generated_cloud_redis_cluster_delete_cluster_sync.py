from google.cloud import redis_cluster_v1

def sample_delete_cluster():
    if False:
        i = 10
        return i + 15
    client = redis_cluster_v1.CloudRedisClusterClient()
    request = redis_cluster_v1.DeleteClusterRequest(name='name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)