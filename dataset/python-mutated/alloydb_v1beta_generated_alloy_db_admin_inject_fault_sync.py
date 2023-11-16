from google.cloud import alloydb_v1beta

def sample_inject_fault():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.InjectFaultRequest(fault_type='STOP_VM', name='name_value')
    operation = client.inject_fault(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)