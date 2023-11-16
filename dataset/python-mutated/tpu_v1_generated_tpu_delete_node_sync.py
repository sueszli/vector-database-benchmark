from google.cloud import tpu_v1

def sample_delete_node():
    if False:
        return 10
    client = tpu_v1.TpuClient()
    request = tpu_v1.DeleteNodeRequest(name='name_value')
    operation = client.delete_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)