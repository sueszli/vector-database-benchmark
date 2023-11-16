from google.cloud import netapp_v1

def sample_create_active_directory():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    active_directory = netapp_v1.ActiveDirectory()
    active_directory.domain = 'domain_value'
    active_directory.dns = 'dns_value'
    active_directory.net_bios_prefix = 'net_bios_prefix_value'
    active_directory.username = 'username_value'
    active_directory.password = 'password_value'
    request = netapp_v1.CreateActiveDirectoryRequest(parent='parent_value', active_directory=active_directory, active_directory_id='active_directory_id_value')
    operation = client.create_active_directory(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)