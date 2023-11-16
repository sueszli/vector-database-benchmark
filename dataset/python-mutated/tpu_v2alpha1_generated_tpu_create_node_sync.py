from google.cloud import tpu_v2alpha1

def sample_create_node():
    if False:
        while True:
            i = 10
    client = tpu_v2alpha1.TpuClient()
    node = tpu_v2alpha1.Node()
    node.runtime_version = 'runtime_version_value'
    request = tpu_v2alpha1.CreateNodeRequest(parent='parent_value', node=node)
    operation = client.create_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)