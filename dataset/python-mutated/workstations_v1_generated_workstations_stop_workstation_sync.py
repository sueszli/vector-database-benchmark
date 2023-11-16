from google.cloud import workstations_v1

def sample_stop_workstation():
    if False:
        return 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.StopWorkstationRequest(name='name_value')
    operation = client.stop_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)