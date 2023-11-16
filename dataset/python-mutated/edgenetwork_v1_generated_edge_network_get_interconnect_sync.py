from google.cloud import edgenetwork_v1

def sample_get_interconnect():
    if False:
        return 10
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.GetInterconnectRequest(name='name_value')
    response = client.get_interconnect(request=request)
    print(response)