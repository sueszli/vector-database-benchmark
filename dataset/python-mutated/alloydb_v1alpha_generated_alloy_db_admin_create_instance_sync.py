from google.cloud import alloydb_v1alpha

def sample_create_instance():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    instance = alloydb_v1alpha.Instance()
    instance.instance_type = 'SECONDARY'
    request = alloydb_v1alpha.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)