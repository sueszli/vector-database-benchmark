from google.cloud import redis_cluster_v1beta1

def sample_delete_cluster():
    if False:
        return 10
    client = redis_cluster_v1beta1.CloudRedisClusterClient()
    request = redis_cluster_v1beta1.DeleteClusterRequest(name='name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)