from google.cloud import edgenetwork_v1

def sample_get_zone():
    if False:
        return 10
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.GetZoneRequest(name='name_value')
    response = client.get_zone(request=request)
    print(response)