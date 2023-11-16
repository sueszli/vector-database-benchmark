from google.cloud import tpu_v2

def sample_update_node():
    if False:
        while True:
            i = 10
    client = tpu_v2.TpuClient()
    node = tpu_v2.Node()
    node.runtime_version = 'runtime_version_value'
    request = tpu_v2.UpdateNodeRequest(node=node)
    operation = client.update_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)