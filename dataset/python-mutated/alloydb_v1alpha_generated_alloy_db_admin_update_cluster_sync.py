from google.cloud import alloydb_v1alpha

def sample_update_cluster():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    cluster = alloydb_v1alpha.Cluster()
    cluster.backup_source.backup_name = 'backup_name_value'
    cluster.network = 'network_value'
    request = alloydb_v1alpha.UpdateClusterRequest(cluster=cluster)
    operation = client.update_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)