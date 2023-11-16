from google.cloud import alloydb_v1alpha

def sample_failover_instance():
    if False:
        return 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.FailoverInstanceRequest(name='name_value')
    operation = client.failover_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)