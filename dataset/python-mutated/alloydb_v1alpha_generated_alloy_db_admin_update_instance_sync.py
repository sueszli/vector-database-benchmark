from google.cloud import alloydb_v1alpha

def sample_update_instance():
    if False:
        return 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    instance = alloydb_v1alpha.Instance()
    instance.instance_type = 'SECONDARY'
    request = alloydb_v1alpha.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)