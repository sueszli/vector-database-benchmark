from google.cloud import netapp_v1

def sample_stop_replication():
    if False:
        print('Hello World!')
    client = netapp_v1.NetAppClient()
    request = netapp_v1.StopReplicationRequest(name='name_value')
    operation = client.stop_replication(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)