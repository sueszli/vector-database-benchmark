from google.cloud import tpu_v2

def sample_create_node():
    if False:
        i = 10
        return i + 15
    client = tpu_v2.TpuClient()
    node = tpu_v2.Node()
    node.runtime_version = 'runtime_version_value'
    request = tpu_v2.CreateNodeRequest(parent='parent_value', node=node)
    operation = client.create_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)