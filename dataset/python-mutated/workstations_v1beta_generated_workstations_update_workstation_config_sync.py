from google.cloud import workstations_v1beta

def sample_update_workstation_config():
    if False:
        i = 10
        return i + 15
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.UpdateWorkstationConfigRequest()
    operation = client.update_workstation_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)