from google.cloud import vmwareengine_v1

def sample_create_hcx_activation_key():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.CreateHcxActivationKeyRequest(parent='parent_value', hcx_activation_key_id='hcx_activation_key_id_value')
    operation = client.create_hcx_activation_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)