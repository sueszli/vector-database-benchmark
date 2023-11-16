from google.cloud import alloydb_v1beta

def sample_update_instance():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1beta.AlloyDBAdminClient()
    instance = alloydb_v1beta.Instance()
    instance.instance_type = 'SECONDARY'
    request = alloydb_v1beta.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)