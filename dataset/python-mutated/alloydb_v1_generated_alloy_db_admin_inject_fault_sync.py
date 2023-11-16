from google.cloud import alloydb_v1

def sample_inject_fault():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.InjectFaultRequest(fault_type='STOP_VM', name='name_value')
    operation = client.inject_fault(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)