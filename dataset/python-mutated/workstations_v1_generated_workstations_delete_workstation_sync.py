from google.cloud import workstations_v1

def sample_delete_workstation():
    if False:
        for i in range(10):
            print('nop')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.DeleteWorkstationRequest(name='name_value')
    operation = client.delete_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)