from google.cloud import alloydb_v1beta

def sample_create_secondary_instance():
    if False:
        print('Hello World!')
    client = alloydb_v1beta.AlloyDBAdminClient()
    instance = alloydb_v1beta.Instance()
    instance.instance_type = 'SECONDARY'
    request = alloydb_v1beta.CreateSecondaryInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_secondary_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)