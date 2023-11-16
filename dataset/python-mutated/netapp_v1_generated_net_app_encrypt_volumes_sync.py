from google.cloud import netapp_v1

def sample_encrypt_volumes():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.EncryptVolumesRequest(name='name_value')
    operation = client.encrypt_volumes(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)