from google.cloud import workstations_v1beta

def sample_stop_workstation():
    if False:
        for i in range(10):
            print('nop')
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.StopWorkstationRequest(name='name_value')
    operation = client.stop_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)