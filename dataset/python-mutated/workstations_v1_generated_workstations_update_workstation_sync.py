from google.cloud import workstations_v1

def sample_update_workstation():
    if False:
        while True:
            i = 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.UpdateWorkstationRequest()
    operation = client.update_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)