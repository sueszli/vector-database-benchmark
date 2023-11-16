from google.cloud import tpu_v1

def sample_stop_node():
    if False:
        for i in range(10):
            print('nop')
    client = tpu_v1.TpuClient()
    request = tpu_v1.StopNodeRequest()
    operation = client.stop_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)