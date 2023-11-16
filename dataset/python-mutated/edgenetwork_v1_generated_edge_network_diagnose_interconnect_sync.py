from google.cloud import edgenetwork_v1

def sample_diagnose_interconnect():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DiagnoseInterconnectRequest(name='name_value')
    response = client.diagnose_interconnect(request=request)
    print(response)