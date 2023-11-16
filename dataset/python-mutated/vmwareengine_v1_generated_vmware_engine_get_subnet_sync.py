from google.cloud import vmwareengine_v1

def sample_get_subnet():
    if False:
        while True:
            i = 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetSubnetRequest(name='name_value')
    response = client.get_subnet(request=request)
    print(response)