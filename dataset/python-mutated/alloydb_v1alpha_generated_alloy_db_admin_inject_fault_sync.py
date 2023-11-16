from google.cloud import alloydb_v1alpha

def sample_inject_fault():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.InjectFaultRequest(fault_type='STOP_VM', name='name_value')
    operation = client.inject_fault(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)