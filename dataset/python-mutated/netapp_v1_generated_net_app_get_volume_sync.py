from google.cloud import netapp_v1

def sample_get_volume():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.GetVolumeRequest(name='name_value')
    response = client.get_volume(request=request)
    print(response)