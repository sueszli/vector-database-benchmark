from google.cloud import edgecontainer_v1

def sample_get_machine():
    if False:
        return 10
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.GetMachineRequest(name='name_value')
    response = client.get_machine(request=request)
    print(response)