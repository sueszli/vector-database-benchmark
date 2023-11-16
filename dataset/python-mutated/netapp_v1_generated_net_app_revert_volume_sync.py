from google.cloud import netapp_v1

def sample_revert_volume():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.RevertVolumeRequest(name='name_value', snapshot_id='snapshot_id_value')
    operation = client.revert_volume(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)