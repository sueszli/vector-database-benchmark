from google.cloud import netapp_v1

def sample_reverse_replication_direction():
    if False:
        print('Hello World!')
    client = netapp_v1.NetAppClient()
    request = netapp_v1.ReverseReplicationDirectionRequest(name='name_value')
    operation = client.reverse_replication_direction(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)