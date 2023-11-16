from google.cloud import netapp_v1

def sample_get_active_directory():
    if False:
        print('Hello World!')
    client = netapp_v1.NetAppClient()
    request = netapp_v1.GetActiveDirectoryRequest(name='name_value')
    response = client.get_active_directory(request=request)
    print(response)