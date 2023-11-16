from google.cloud import workstations_v1

def sample_get_workstation():
    if False:
        print('Hello World!')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.GetWorkstationRequest(name='name_value')
    response = client.get_workstation(request=request)
    print(response)