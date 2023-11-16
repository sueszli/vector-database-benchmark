from google.cloud import netapp_v1

def sample_delete_active_directory():
    if False:
        for i in range(10):
            print('nop')
    client = netapp_v1.NetAppClient()
    request = netapp_v1.DeleteActiveDirectoryRequest(name='name_value')
    operation = client.delete_active_directory(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)