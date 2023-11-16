from google.cloud import edgenetwork_v1

def sample_get_subnet():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.GetSubnetRequest(name='name_value')
    response = client.get_subnet(request=request)
    print(response)