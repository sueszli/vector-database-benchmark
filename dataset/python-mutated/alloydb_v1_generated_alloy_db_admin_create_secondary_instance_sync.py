from google.cloud import alloydb_v1

def sample_create_secondary_instance():
    if False:
        while True:
            i = 10
    client = alloydb_v1.AlloyDBAdminClient()
    instance = alloydb_v1.Instance()
    instance.instance_type = 'SECONDARY'
    request = alloydb_v1.CreateSecondaryInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_secondary_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)