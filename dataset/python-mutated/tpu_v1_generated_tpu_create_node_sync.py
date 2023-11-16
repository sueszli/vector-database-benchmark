from google.cloud import tpu_v1

def sample_create_node():
    if False:
        i = 10
        return i + 15
    client = tpu_v1.TpuClient()
    node = tpu_v1.Node()
    node.accelerator_type = 'accelerator_type_value'
    node.tensorflow_version = 'tensorflow_version_value'
    request = tpu_v1.CreateNodeRequest(parent='parent_value', node=node)
    operation = client.create_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)