from google.cloud import edgenetwork_v1

def sample_diagnose_router():
    if False:
        while True:
            i = 10
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DiagnoseRouterRequest(name='name_value')
    response = client.diagnose_router(request=request)
    print(response)