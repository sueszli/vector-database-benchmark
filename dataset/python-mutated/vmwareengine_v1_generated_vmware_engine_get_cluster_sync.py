from google.cloud import vmwareengine_v1

def sample_get_cluster():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)