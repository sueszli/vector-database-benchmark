from google.cloud import edgenetwork_v1

def sample_get_network():
    if False:
        i = 10
        return i + 15
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.GetNetworkRequest(name='name_value')
    response = client.get_network(request=request)
    print(response)