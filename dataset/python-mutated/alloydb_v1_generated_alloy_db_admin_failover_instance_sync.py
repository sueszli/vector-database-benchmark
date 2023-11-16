from google.cloud import alloydb_v1

def sample_failover_instance():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.FailoverInstanceRequest(name='name_value')
    operation = client.failover_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)