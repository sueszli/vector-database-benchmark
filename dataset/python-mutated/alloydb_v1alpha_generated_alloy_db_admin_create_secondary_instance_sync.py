from google.cloud import alloydb_v1alpha

def sample_create_secondary_instance():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1alpha.AlloyDBAdminClient()
    instance = alloydb_v1alpha.Instance()
    instance.instance_type = 'SECONDARY'
    request = alloydb_v1alpha.CreateSecondaryInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_secondary_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)