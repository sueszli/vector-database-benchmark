from google.cloud import redis_cluster_v1beta1

def sample_update_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = redis_cluster_v1beta1.CloudRedisClusterClient()
    cluster = redis_cluster_v1beta1.Cluster()
    cluster.name = 'name_value'
    cluster.psc_configs.network = 'network_value'
    request = redis_cluster_v1beta1.UpdateClusterRequest(cluster=cluster)
    operation = client.update_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)