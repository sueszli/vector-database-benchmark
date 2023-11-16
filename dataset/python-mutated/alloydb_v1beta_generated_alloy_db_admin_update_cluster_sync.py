from google.cloud import alloydb_v1beta

def sample_update_cluster():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    cluster = alloydb_v1beta.Cluster()
    cluster.backup_source.backup_name = 'backup_name_value'
    cluster.network = 'network_value'
    request = alloydb_v1beta.UpdateClusterRequest(cluster=cluster)
    operation = client.update_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)