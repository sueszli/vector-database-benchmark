from google.cloud import netapp_v1

def sample_update_replication():
    if False:
        return 10
    client = netapp_v1.NetAppClient()
    replication = netapp_v1.Replication()
    replication.replication_schedule = 'DAILY'
    replication.destination_volume_parameters.storage_pool = 'storage_pool_value'
    request = netapp_v1.UpdateReplicationRequest(replication=replication)
    operation = client.update_replication(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)