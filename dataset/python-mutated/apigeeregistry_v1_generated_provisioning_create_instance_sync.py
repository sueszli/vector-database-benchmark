from google.cloud import apigee_registry_v1

def sample_create_instance():
    if False:
        for i in range(10):
            print('nop')
    client = apigee_registry_v1.ProvisioningClient()
    instance = apigee_registry_v1.Instance()
    instance.config.cmek_key_name = 'cmek_key_name_value'
    request = apigee_registry_v1.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)