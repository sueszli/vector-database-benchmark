from google.cloud import edgenetwork_v1

def sample_get_router():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.GetRouterRequest(name='name_value')
    response = client.get_router(request=request)
    print(response)