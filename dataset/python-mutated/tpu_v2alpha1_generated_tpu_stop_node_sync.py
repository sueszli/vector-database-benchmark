from google.cloud import tpu_v2alpha1

def sample_stop_node():
    if False:
        return 10
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.StopNodeRequest(name='name_value')
    operation = client.stop_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)