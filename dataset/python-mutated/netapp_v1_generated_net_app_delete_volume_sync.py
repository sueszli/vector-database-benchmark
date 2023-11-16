from google.cloud import netapp_v1

def sample_delete_volume():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.DeleteVolumeRequest(name='name_value')
    operation = client.delete_volume(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)