from google.cloud import netapp_v1

def sample_get_replication():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.GetReplicationRequest(name='name_value')
    response = client.get_replication(request=request)
    print(response)