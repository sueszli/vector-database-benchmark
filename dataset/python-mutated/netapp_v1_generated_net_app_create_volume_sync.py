from google.cloud import netapp_v1

def sample_create_volume():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    volume = netapp_v1.Volume()
    volume.share_name = 'share_name_value'
    volume.storage_pool = 'storage_pool_value'
    volume.capacity_gib = 1247
    volume.protocols = ['SMB']
    request = netapp_v1.CreateVolumeRequest(parent='parent_value', volume_id='volume_id_value', volume=volume)
    operation = client.create_volume(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)