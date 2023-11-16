from google.cloud import edgenetwork_v1

def sample_diagnose_network():
    if False:
        while True:
            i = 10
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DiagnoseNetworkRequest(name='name_value')
    response = client.diagnose_network(request=request)
    print(response)