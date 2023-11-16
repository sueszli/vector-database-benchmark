from google.cloud import netapp_v1

def sample_delete_replication():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    request = netapp_v1.DeleteReplicationRequest(name='name_value')
    operation = client.delete_replication(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)