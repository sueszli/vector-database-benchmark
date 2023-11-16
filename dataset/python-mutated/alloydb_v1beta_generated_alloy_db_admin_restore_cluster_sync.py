from google.cloud import alloydb_v1beta

def sample_restore_cluster():
    if False:
        return 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    backup_source = alloydb_v1beta.BackupSource()
    backup_source.backup_name = 'backup_name_value'
    cluster = alloydb_v1beta.Cluster()
    cluster.backup_source.backup_name = 'backup_name_value'
    cluster.network = 'network_value'
    request = alloydb_v1beta.RestoreClusterRequest(backup_source=backup_source, parent='parent_value', cluster_id='cluster_id_value', cluster=cluster)
    operation = client.restore_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)