from google.cloud import workstations_v1beta

def sample_delete_workstation():
    if False:
        i = 10
        return i + 15
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.DeleteWorkstationRequest(name='name_value')
    operation = client.delete_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)