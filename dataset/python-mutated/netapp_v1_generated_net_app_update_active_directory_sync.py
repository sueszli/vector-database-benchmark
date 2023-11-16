from google.cloud import netapp_v1

def sample_update_active_directory():
    if False:
        return 10
    client = netapp_v1.NetAppClient()
    active_directory = netapp_v1.ActiveDirectory()
    active_directory.domain = 'domain_value'
    active_directory.dns = 'dns_value'
    active_directory.net_bios_prefix = 'net_bios_prefix_value'
    active_directory.username = 'username_value'
    active_directory.password = 'password_value'
    request = netapp_v1.UpdateActiveDirectoryRequest(active_directory=active_directory)
    operation = client.update_active_directory(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)