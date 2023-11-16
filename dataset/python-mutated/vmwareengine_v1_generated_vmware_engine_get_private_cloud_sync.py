from google.cloud import vmwareengine_v1

def sample_get_private_cloud():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetPrivateCloudRequest(name='name_value')
    response = client.get_private_cloud(request=request)
    print(response)