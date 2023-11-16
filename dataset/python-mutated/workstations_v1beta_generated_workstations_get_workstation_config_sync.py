from google.cloud import workstations_v1beta

def sample_get_workstation_config():
    if False:
        print('Hello World!')
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.GetWorkstationConfigRequest(name='name_value')
    response = client.get_workstation_config(request=request)
    print(response)