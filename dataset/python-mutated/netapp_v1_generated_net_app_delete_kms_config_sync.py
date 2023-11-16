from google.cloud import netapp_v1

def sample_delete_kms_config():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.DeleteKmsConfigRequest(name='name_value')
    operation = client.delete_kms_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)