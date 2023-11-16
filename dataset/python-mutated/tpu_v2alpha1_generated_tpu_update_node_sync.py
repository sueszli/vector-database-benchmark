from google.cloud import tpu_v2alpha1

def sample_update_node():
    if False:
        for i in range(10):
            print('nop')
    client = tpu_v2alpha1.TpuClient()
    node = tpu_v2alpha1.Node()
    node.runtime_version = 'runtime_version_value'
    request = tpu_v2alpha1.UpdateNodeRequest(node=node)
    operation = client.update_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)