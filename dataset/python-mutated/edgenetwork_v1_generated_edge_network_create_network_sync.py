from google.cloud import edgenetwork_v1

def sample_create_network():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    network = edgenetwork_v1.Network()
    network.name = 'name_value'
    request = edgenetwork_v1.CreateNetworkRequest(parent='parent_value', network_id='network_id_value', network=network)
    operation = client.create_network(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)