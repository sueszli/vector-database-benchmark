from google.cloud import netapp_v1

def sample_create_replication():
    if False:
        print('Hello World!')
    client = netapp_v1.NetAppClient()
    replication = netapp_v1.Replication()
    replication.replication_schedule = 'DAILY'
    replication.destination_volume_parameters.storage_pool = 'storage_pool_value'
    request = netapp_v1.CreateReplicationRequest(parent='parent_value', replication=replication, replication_id='replication_id_value')
    operation = client.create_replication(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)