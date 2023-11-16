from google.cloud import alloydb_v1

def sample_create_secondary_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1.AlloyDBAdminClient()
    cluster = alloydb_v1.Cluster()
    cluster.backup_source.backup_name = 'backup_name_value'
    cluster.network = 'network_value'
    request = alloydb_v1.CreateSecondaryClusterRequest(parent='parent_value', cluster_id='cluster_id_value', cluster=cluster)
    operation = client.create_secondary_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)