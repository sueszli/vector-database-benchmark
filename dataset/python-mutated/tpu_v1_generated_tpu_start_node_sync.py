from google.cloud import tpu_v1

def sample_start_node():
    if False:
        print('Hello World!')
    client = tpu_v1.TpuClient()
    request = tpu_v1.StartNodeRequest()
    operation = client.start_node(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)