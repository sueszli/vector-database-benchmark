from google.cloud import workstations_v1beta

def sample_update_workstation():
    if False:
        return 10
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.UpdateWorkstationRequest()
    operation = client.update_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)