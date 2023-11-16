from google.cloud import vmwareengine_v1

def sample_get_private_connection():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetPrivateConnectionRequest(name='name_value')
    response = client.get_private_connection(request=request)
    print(response)