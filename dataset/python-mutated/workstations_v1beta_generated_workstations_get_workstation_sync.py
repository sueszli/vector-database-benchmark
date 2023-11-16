from google.cloud import workstations_v1beta

def sample_get_workstation():
    if False:
        i = 10
        return i + 15
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.GetWorkstationRequest(name='name_value')
    response = client.get_workstation(request=request)
    print(response)