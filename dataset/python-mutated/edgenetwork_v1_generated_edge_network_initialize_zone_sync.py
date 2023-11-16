from google.cloud import edgenetwork_v1

def sample_initialize_zone():
    if False:
        print('Hello World!')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.InitializeZoneRequest(name='name_value')
    response = client.initialize_zone(request=request)
    print(response)