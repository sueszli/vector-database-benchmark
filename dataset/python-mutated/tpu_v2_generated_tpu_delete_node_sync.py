from google.cloud import tpu_v2

def sample_delete_node():
    if False:
        i = 10
        return i + 15
    client = tpu_v2.TpuClient()
    request = tpu_v2.DeleteNodeRequest(name='name_value')
    operation = client.delete_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)