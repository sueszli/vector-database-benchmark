from google.cloud import edgenetwork_v1

def sample_update_subnet():
    if False:
        while True:
            i = 10
    client = edgenetwork_v1.EdgeNetworkClient()
    subnet = edgenetwork_v1.Subnet()
    subnet.name = 'name_value'
    subnet.network = 'network_value'
    request = edgenetwork_v1.UpdateSubnetRequest(subnet=subnet)
    operation = client.update_subnet(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)