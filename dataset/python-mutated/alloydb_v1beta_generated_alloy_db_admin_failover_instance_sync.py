from google.cloud import alloydb_v1beta

def sample_failover_instance():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.FailoverInstanceRequest(name='name_value')
    operation = client.failover_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)