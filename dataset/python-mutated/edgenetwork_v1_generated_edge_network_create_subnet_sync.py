from google.cloud import edgenetwork_v1

def sample_create_subnet():
    if False:
        print('Hello World!')
    client = edgenetwork_v1.EdgeNetworkClient()
    subnet = edgenetwork_v1.Subnet()
    subnet.name = 'name_value'
    subnet.network = 'network_value'
    request = edgenetwork_v1.CreateSubnetRequest(parent='parent_value', subnet_id='subnet_id_value', subnet=subnet)
    operation = client.create_subnet(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)