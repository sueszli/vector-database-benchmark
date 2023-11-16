from google.cloud import vmwareengine_v1

def sample_delete_cluster():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeleteClusterRequest(name='name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)