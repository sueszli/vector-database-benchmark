from google.cloud import vmmigration_v1

def sample_get_replication_cycle():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetReplicationCycleRequest(name='name_value')
    response = client.get_replication_cycle(request=request)
    print(response)