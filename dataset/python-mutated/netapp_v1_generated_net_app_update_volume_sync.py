from google.cloud import netapp_v1

def sample_update_volume():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    volume = netapp_v1.Volume()
    volume.share_name = 'share_name_value'
    volume.storage_pool = 'storage_pool_value'
    volume.capacity_gib = 1247
    volume.protocols = ['SMB']
    request = netapp_v1.UpdateVolumeRequest(volume=volume)
    operation = client.update_volume(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)