from google.cloud import vmwareengine_v1

def sample_create_cluster():
    if False:
        while True:
            i = 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.CreateClusterRequest(parent='parent_value', cluster_id='cluster_id_value')
    operation = client.create_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)