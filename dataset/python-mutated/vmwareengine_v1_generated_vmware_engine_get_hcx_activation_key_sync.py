from google.cloud import vmwareengine_v1

def sample_get_hcx_activation_key():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetHcxActivationKeyRequest(name='name_value')
    response = client.get_hcx_activation_key(request=request)
    print(response)