from google.cloud import vmwareengine_v1

def sample_update_private_connection():
    if False:
        return 10
    client = vmwareengine_v1.VmwareEngineClient()
    private_connection = vmwareengine_v1.PrivateConnection()
    private_connection.vmware_engine_network = 'vmware_engine_network_value'
    private_connection.type_ = 'THIRD_PARTY_SERVICE'
    private_connection.service_network = 'service_network_value'
    request = vmwareengine_v1.UpdatePrivateConnectionRequest(private_connection=private_connection)
    operation = client.update_private_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)