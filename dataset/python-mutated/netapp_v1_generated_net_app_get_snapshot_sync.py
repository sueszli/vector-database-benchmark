from google.cloud import netapp_v1

def sample_get_snapshot():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    request = netapp_v1.GetSnapshotRequest(name='name_value')
    response = client.get_snapshot(request=request)
    print(response)